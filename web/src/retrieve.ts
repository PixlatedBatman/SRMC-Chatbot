import corpus from "./corpus.json";

export const TOP_K = 6;

export interface ChunkMeta {
  id: string;
  source_type: string;
  source_name?: string;
  authority: number;
  has_solution?: boolean;
  exam?: string;
  year?: number;
  question_number?: string;
  section_number?: string;
  section_title?: string;
  source_url?: string;
}

export interface Chunk {
  text: string;
  meta: ChunkMeta;
}

export interface ScoredChunk extends Chunk {
  score: number;
}

const chunks = corpus.chunks as unknown as Chunk[];
const dim = corpus.dim;

// Decoded once per isolate, then reused across requests. 87 x 384 float32 is
// ~130KB resident, so there is no reason to be lazier than this.
const matrix = decodeFloat32(corpus.embeddings);

if (matrix.length !== chunks.length * dim) {
  throw new Error(
    `corpus.json is inconsistent: ${matrix.length} floats for ${chunks.length} chunks x ${dim} dims`,
  );
}

function decodeFloat32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const norm = Math.sqrt(sum);
  // A zero vector would mean the embedding call returned garbage; leaving it
  // unnormalized scores everything 0, which is a more honest failure than NaN.
  if (norm === 0) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/**
 * Embed the question with Workers AI. The BGE query-side instruction prefix is
 * applied here and only here - corpus vectors were built from bare passages.
 */
export async function embedQuery(ai: Ai, question: string): Promise<Float32Array> {
  // Model name comes from corpus.json, so it isn't a literal `keyof AiModels`
  // and `ai.run` needs a type cast. Cast `ai` itself, not the extracted
  // method - `Ai.run` is a class method that keeps state in private fields,
  // so calling it detached from `ai` (e.g. `const run = ai.run; run(...)`)
  // loses the `this` binding and throws inside the private-field access.
  // Casting `ai` and calling `.run(...)` on it keeps the call a method call.
  const looselyTyped = ai as unknown as { run(model: string, inputs: unknown): Promise<unknown> };
  const res = (await looselyTyped.run(corpus.embedModel, {
    text: [corpus.queryPrefix + question],
  })) as { data?: number[][] };

  const vec = res?.data?.[0];
  if (!Array.isArray(vec) || vec.length !== dim) {
    throw new Error(`Embedding model returned ${vec?.length ?? "no"} dims, expected ${dim}`);
  }
  // Workers AI does not promise normalized output, and the corpus side is
  // normalized, so cosine == dot only after we normalize here.
  return normalize(Float32Array.from(vec));
}

/**
 * Brute-force cosine over the whole corpus. At 87 chunks this is ~33k
 * multiply-adds - far cheaper than the network hop a vector DB would add.
 *
 * Results are sorted by authority first (rulebook > website > pyq), then by
 * similarity within each tier. That ordering is what makes rule 2 of the system
 * prompt work: the model sees the most authoritative chunks first.
 */
export function search(query: Float32Array, k: number = TOP_K): ScoredChunk[] {
  const scored: ScoredChunk[] = chunks.map((chunk, i) => {
    const offset = i * dim;
    let dot = 0;
    for (let j = 0; j < dim; j++) dot += query[j] * matrix[offset + j];
    return { ...chunk, score: dot };
  });

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, k);
  top.sort((a, b) => (a.meta.authority ?? 99) - (b.meta.authority ?? 99) || b.score - a.score);
  return top;
}

export function buildContext(top: ScoredChunk[]): string {
  return top
    .map((c, i) => {
      const m = c.meta;
      const parts = [
        `Source ${i + 1}`,
        m.source_type,
        `authority=${m.authority}`,
        m.source_name ?? m.id,
      ];
      // Only PYQ records carry has_solution. Surfacing it is what lets the model
      // actually apply rule 3 ("no official solution available") instead of
      // being told about a field it never sees.
      if (typeof m.has_solution === "boolean") parts.push(`has_solution=${m.has_solution}`);
      return `[${parts.join(" | ")}]\n${c.text}`;
    })
    .join("\n\n");
}

export function toSources(top: ScoredChunk[]) {
  return top.map((c) => ({
    source_name: c.meta.source_name ?? c.meta.id,
    authority: c.meta.authority,
    source_url: c.meta.source_url,
  }));
}
