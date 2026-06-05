"""Build embedding + FAISS + BM25 indices over chunked corpus.

Upgrades over baseline:
- BM25 index alongside FAISS for hybrid sparse + dense retrieval
- Same MiniLM embedder, IndexFlatIP (cosine) since 133 chunks is tiny
"""
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    """Simple lowercase whitespace tokenizer for BM25."""
    return [t for t in text.lower().split() if t.isalnum() or any(c.isalnum() for c in t)]


def main() -> None:
    chunks = pd.read_pickle(ROOT / "data" / "chunks.pkl")
    texts = chunks["text"].tolist()
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")

    embedder = SentenceTransformer(EMBED_MODEL)
    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    dim = emb.shape[1]
    print(f"Embeddings: {emb.shape}")

    # FAISS: inner-product index over normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    print(f"FAISS IndexFlatIP saved (dim={dim}, ntotal={index.ntotal})")

    # BM25: lowercase whitespace tokenizer over chunk text
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    print(f"BM25 index saved ({len(tokenized)} docs)")

    chunks.to_pickle(INDEX_DIR / "chunks.pkl")
    print(f"Chunks pickle saved → {INDEX_DIR / 'chunks.pkl'}")


if __name__ == "__main__":
    main()
