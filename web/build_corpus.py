"""
One-time build step: turn unified_dataset.json into src/corpus.json for the Worker.

Run this whenever the dataset changes, then redeploy:

    pip install sentence-transformers
    python3 build_corpus.py

Why this exists
---------------
The Worker can't run sentence-transformers (no PyTorch in a Workers isolate),
so it embeds the *query* at request time via Workers AI and needs the *corpus*
embeddings precomputed here. Both sides must use the same model or retrieval
silently degrades into nonsense - see EMBED_MODEL below.

Model choice
------------
The original index used all-MiniLM-L6-v2, which Workers AI does not host.
We switch to BAAI/bge-small-en-v1.5, which Workers AI *does* host as
'@cf/baai/bge-small-en-v1.5'. Same 384 dims, better retrieval quality.
This script runs the model locally from HuggingFace; Workers AI runs the same
weights server-side. Vectors agree to float precision, which is far below what
would reorder a top-6 out of 87 chunks.

BGE asymmetry
-------------
BGE models are trained with an instruction prefix on the *query* side only.
Passages are embedded bare (here); queries get QUERY_PREFIX prepended (in the
Worker). Getting this backwards, or applying it to both, measurably hurts
recall - the constant is duplicated into corpus.json so the two sides can't
drift apart.
"""

import base64
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
WORKERS_AI_MODEL = "@cf/baai/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DATASET = Path(__file__).parent.parent / "data" / "unified_dataset.json"
OUT = Path(__file__).parent / "src" / "corpus.json"

# Fields worth carrying into the Worker: everything the prompt context header
# or the UI's source list actually reads. Big text fields are excluded - the
# retrieval unit is chunk_text and duplicating it into metadata just inflates
# the bundle.
META_KEYS = [
    "id", "source_type", "source_name", "authority",
    "exam", "year", "question_number", "has_solution",
    "section_number", "section_title", "source_url",
]


def main() -> None:
    records = json.loads(DATASET.read_text())
    for r in records:
        if not r.get("chunk_text", "").strip():
            raise ValueError(f"Record {r.get('id')} has no chunk_text")
    print(f"Loaded {len(records)} chunks from {DATASET}")

    print(f"Loading {EMBED_MODEL} (first run downloads ~130MB)")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [r["chunk_text"] for r in records]
    print(f"Embedding {len(texts)} passages (no query prefix - passage side)")
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="<f4")

    if emb.shape != (len(records), 384):
        raise ValueError(f"Unexpected embedding shape {emb.shape}, expected ({len(records)}, 384)")

    chunks = []
    for r in records:
        meta = {k: r[k] for k in META_KEYS if k in r and r[k] is not None}
        chunks.append({"text": r["chunk_text"], "meta": meta})

    payload = {
        "embedModel": WORKERS_AI_MODEL,
        "dim": int(emb.shape[1]),
        "queryPrefix": QUERY_PREFIX,
        # Row-major float32, normalized. Base64 keeps the bundle ~180KB instead
        # of ~330KB of JSON number literals, and decodes in one pass.
        "embeddings": base64.b64encode(emb.tobytes()).decode("ascii"),
        "chunks": chunks,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(chunks)} chunks)")


if __name__ == "__main__":
    main()
