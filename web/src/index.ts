import Anthropic from "@anthropic-ai/sdk";
import { SYSTEM_PROMPT } from "./prompt";
import { buildContext, embedQuery, search, toSources } from "./retrieve";

export interface Env {
  AI: Ai;
  ASSETS: Fetcher;
  ANTHROPIC_API_KEY: string;
  CLAUDE_MODEL?: string;
}

const DEFAULT_MODEL = "claude-opus-5";

// Thinking is on by default on Opus 5 and max_tokens caps thinking + text
// together, so this is deliberately well above the 1000 the Python version
// used. Low effort keeps a grounded 6-chunk answer fast without disabling
// thinking outright (which risks <thinking> tags leaking into the reply).
const MAX_TOKENS = 4000;
const EFFORT = "low" as const;

const MAX_TURNS = 20;
const MAX_CHARS_PER_MESSAGE = 4000;

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

class BadRequest extends Error {}

function parseMessages(body: unknown): ChatMessage[] {
  if (typeof body !== "object" || body === null) throw new BadRequest("Body must be a JSON object");
  const raw = (body as { messages?: unknown }).messages;
  if (!Array.isArray(raw) || raw.length === 0) throw new BadRequest("messages must be a non-empty array");
  if (raw.length > MAX_TURNS) throw new BadRequest(`Conversation too long (max ${MAX_TURNS} turns)`);

  const messages = raw.map((m, i): ChatMessage => {
    const role = (m as ChatMessage)?.role;
    const content = (m as ChatMessage)?.content;
    if (role !== "user" && role !== "assistant") throw new BadRequest(`messages[${i}].role must be user or assistant`);
    if (typeof content !== "string" || content.trim() === "") throw new BadRequest(`messages[${i}].content must be a non-empty string`);
    if (content.length > MAX_CHARS_PER_MESSAGE) throw new BadRequest(`messages[${i}].content exceeds ${MAX_CHARS_PER_MESSAGE} characters`);
    return { role, content };
  });

  if (messages[messages.length - 1].role !== "user") throw new BadRequest("Last message must be from the user");
  return messages;
}

function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * console.error("label", err) renders unreliably in Workers Logs - observed
 * cases where only stack frames show and the error's own message is dropped
 * entirely (e.g. AiError instances, which carry detail in non-standard own
 * properties rather than a populated `.message`). Dump every own property
 * into one string so nothing gets lost to the platform's formatting.
 */
function describeError(err: unknown): string {
  if (err instanceof Error) {
    const props: Record<string, unknown> = {};
    for (const key of Object.getOwnPropertyNames(err)) {
      if (key === "stack") continue;
      props[key] = (err as unknown as Record<string, unknown>)[key];
    }
    return JSON.stringify(props);
  }
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

async function handleChat(request: Request, env: Env): Promise<Response> {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  if (!env.ANTHROPIC_API_KEY) return Response.json({ error: "Server is missing ANTHROPIC_API_KEY" }, { status: 500 });

  let messages: ChatMessage[];
  try {
    messages = parseMessages(await request.json());
  } catch (err) {
    const message = err instanceof BadRequest ? err.message : "Malformed JSON body";
    return Response.json({ error: message }, { status: 400 });
  }

  const question = messages[messages.length - 1].content;

  // Retrieve before opening the stream so a retrieval failure is a clean JSON
  // error rather than an SSE stream that dies two tokens in.
  let top, context, sources;
  try {
    top = search(await embedQuery(env.AI, question));
    context = buildContext(top);
    sources = toSources(top);
  } catch (err) {
    console.error(`retrieval failed: ${describeError(err)}`);
    return Response.json({ error: "Retrieval failed. Please try again." }, { status: 502 });
  }

  const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
  const encoder = new TextEncoder();

  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: string, data: unknown) => controller.enqueue(encoder.encode(sse(event, data)));
      send("sources", sources);

      try {
        const stream = client.messages.stream({
          model: env.CLAUDE_MODEL || DEFAULT_MODEL,
          max_tokens: MAX_TOKENS,
          output_config: { effort: EFFORT },
          system: SYSTEM_PROMPT,
          messages: [
            ...messages.slice(0, -1),
            { role: "user", content: `Context:\n\n${context}\n\n---\n\nQuestion: ${question}` },
          ],
        });

        for await (const event of stream) {
          if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
            send("delta", { text: event.delta.text });
          }
        }

        const final = await stream.finalMessage();
        if (final.stop_reason === "refusal") {
          send("error", { message: "That request was declined. Try rephrasing your question." });
        } else {
          send("done", { stop_reason: final.stop_reason });
        }
      } catch (err) {
        console.error(`generation failed: ${describeError(err)}`);
        send("error", { message: "The assistant could not finish that answer. Please try again." });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(body, {
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-store",
      connection: "keep-alive",
    },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url);
    if (pathname === "/api/chat") return handleChat(request, env);
    if (pathname.startsWith("/api/")) return Response.json({ error: "Not found" }, { status: 404 });
    // Static assets are served ahead of the Worker by default; this is the
    // fallback for anything that slips through.
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
