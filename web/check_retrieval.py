"""
Reference retrieval, computed locally against src/corpus.json.

Use this to sanity-check the deployed Worker: run this script, then ask the
same questions in the live UI and confirm the "Sources used" list matches.
A mismatch almost always means index/query embedding drift - see the notes in
build_corpus.py.

    ../data/rag_env/bin/python check_retrieval.py
    ../data/rag_env/bin/python check_retrieval.py "your own question"
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

TOP_K = 6
DEFAULT_QUESTIONS = [
    "What happens if I get caught cheating?",
    "Is there an age limit to register?",
    "What are the prize amounts?",
]


def main() -> None:
    corpus = json.loads((Path(__file__).parent / "src" / "corpus.json").read_text())
    chunks = corpus["chunks"]
    emb = np.frombuffer(base64.b64decode(corpus["embeddings"]), dtype="<f4")
    emb = emb.reshape(len(chunks), corpus["dim"])

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    questions = sys.argv[1:] or DEFAULT_QUESTIONS

    for question in questions:
        vec = model.encode([corpus["queryPrefix"] + question], normalize_embeddings=True)[0]
        scores = emb @ vec.astype("<f4")

        top = np.argsort(-scores)[:TOP_K]
        # Same two-key ordering as the Worker: authority tier, then similarity.
        top = sorted(top, key=lambda i: (chunks[i]["meta"].get("authority", 99), -scores[i]))

        print(f"\n{question}")
        for i in top:
            meta = chunks[i]["meta"]
            print(
                f"  authority={meta.get('authority')}  score={scores[i]:.3f}  "
                f"{meta.get('source_name', meta['id'])}"
            )


if __name__ == "__main__":
    main()
