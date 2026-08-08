# SRMC Chatbot

A RAG chatbot for the Srinivasa Ramanujan Mathematics Competition (SRMC) that
answers participant questions from the official rulebook, website FAQ, and
past exam questions — rulebook wins on any conflict.

## Layout

```
data/   Local-only: dataset, embedding pipeline, original Streamlit prototype
web/    Deployed app: Cloudflare Worker + static UI
```

**`web/` is what's live.** It's a single Cloudflare Worker that serves the
chat UI and handles retrieval + generation, with no server to manage and no
vector database — see `web/README.md` for the architecture and deploy steps.

**`data/` is local-only and gitignored.** It's where the dataset lives and
where retrieval changes get prototyped before being ported to `web/`:

| File | Purpose |
| --- | --- |
| `unified_dataset.json` | The source corpus (rulebook, FAQ, PYQs) — the one file `web/build_corpus.py` reads |
| `embed_and_index.py` | Builds a local Chroma index from the dataset, for experimenting with retrieval |
| `rag_generate.py` | Retrieve → prompt → generate, runnable from the CLI against that Chroma index |
| `app.py` | The original Streamlit UI (superseded by `web/`, kept for local testing) |
| `helper files/` | One-off scripts used to build `unified_dataset.json` from raw sources |

If you edit the system prompt or retrieval logic, `data/rag_generate.py` is
the fast place to iterate locally (`streamlit run app.py`); once it's right,
port the change to `web/src/prompt.ts` / `web/src/retrieve.ts`, which is the
copy that actually serves users.

## Updating the corpus

Whenever `data/unified_dataset.json` changes:

```bash
cd web
../data/rag_env/bin/python build_corpus.py
git add src/corpus.json
```

Then redeploy (`npm run deploy`, or push if using Cloudflare's Git-connected
build). See `web/README.md` for full deploy instructions and operational
notes (rate limiting, model choice, embedding-model constraints).
