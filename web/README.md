# SRMC Chatbot — Cloudflare deployment

The Streamlit app (`../data/app.py`) rewritten as a static UI plus a single
Cloudflare Worker. No vector database: at 87 chunks the embeddings are ~130KB,
so they ship inside the Worker bundle and cosine similarity runs in a loop.

```
Browser ──> Worker (src/index.ts) ──> Anthropic Messages API
   ↑          │
public/       ├─ Workers AI  @cf/baai/bge-small-en-v1.5  (query embedding)
(Assets)      └─ src/corpus.json                          (87 passage vectors)
```

| Path | What it is |
| --- | --- |
| `build_corpus.py` | One-time: `unified_dataset.json` → `src/corpus.json` |
| `src/index.ts` | HTTP handling, SSE streaming, Anthropic call |
| `src/retrieve.ts` | Cosine search, authority sort, context builder |
| `src/prompt.ts` | System prompt, ported from `rag_generate.py` |
| `public/` | Static chat UI (no framework, no CDN dependencies) |
| `check_retrieval.py` | Local reference results, for verifying the deploy |

## Deploy

```bash
npm install

# 1. Build the corpus (needs sentence-transformers; the repo's venv has it)
../data/rag_env/bin/python build_corpus.py

# 2. Authenticate and set the API key
npx wrangler login
npx wrangler secret put ANTHROPIC_API_KEY

# 3. Ship
npm run deploy
```

Then attach your domain: **Workers & Pages → srmc-chatbot → Settings → Domains
& Routes → Add custom domain**. Cloudflare provisions the certificate and DNS
record automatically.

### Local development

```bash
npm run dev -- --remote
```

`--remote` is required. Workers AI has no local emulation, so `wrangler dev`
without it will fail on the embedding call.

## Things worth knowing

**Re-run `build_corpus.py` whenever the dataset changes.** The Worker will
happily serve a stale corpus — nothing detects drift between
`unified_dataset.json` and `src/corpus.json`.

**The embedding model changed.** The original index used `all-MiniLM-L6-v2`,
which Workers AI does not host. Both sides now use BGE-small (384-dim, same as
before, better retrieval). The index is built locally from HuggingFace and the
query is embedded by Workers AI from the same weights. If you swap the model,
you must change it in **both** places — `build_corpus.py` writes the model name
into `corpus.json` and the Worker reads it from there, so they cannot silently
diverge, but a mismatch in weights would still degrade retrieval quietly.
Verify with `check_retrieval.py` after any change.

**BGE uses an asymmetric prefix.** Queries get
`"Represent this sentence for searching relevant passages: "`; passages do not.
This is handled in `retrieve.ts:embedQuery` and nowhere else.

**`ANTHROPIC_API_KEY` is a Worker secret and never reaches the browser.** There
is deliberately no CORS header on `/api/chat`, so only your own origin can call
it. That is not a substitute for rate limiting — `/api/chat` is a public
endpoint that spends money. Before going live, add a rate limiting rule:
**Security → WAF → Rate limiting rules**, matching `URI Path eq /api/chat`, e.g.
10 requests per minute per IP. The Worker caps message length (4000 chars) and
history depth (20 turns) but cannot see request rates.

**Model.** Defaults to `claude-opus-5` via the `CLAUDE_MODEL` var in
`wrangler.jsonc`. Set it to `claude-sonnet-4-6` to match the original Python
version. Thinking is on (adaptive) at `effort: "low"`, with `max_tokens: 4000` —
on Opus 5 `max_tokens` bounds thinking *and* response text together, so the
1000 the Python version used would truncate answers.

**LaTeX is not typeset.** Past-paper questions contain `$...$` math, which
renders as literal text in both the old Streamlit app and this one. Adding
KaTeX to `public/` would fix it; it is not wired up.

## Costs

Workers, Workers AI, and Assets all sit inside the free tiers at this volume.
Anthropic API usage is the only real cost.

## What still runs in Python

Nothing in the request path. `../data/embed_and_index.py` and
`../data/rag_generate.py` remain useful for local retrieval experiments against
Chroma, but they are no longer what serves traffic. If you change the system
prompt, change it in `src/prompt.ts` — that is the copy the Worker uses.
