"""Hybrid retrieval: BM25 + Dense via Reciprocal Rank Fusion + optional reranker.

Upgrades over baseline:
- Dense-only cosine → BM25 sparse + Dense via RRF (captures both keyword and semantic)
- Optional cross-encoder reranker over fused top-20 to produce final top-5
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # public, light, free


@dataclass
class Hit:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    score: float
    source: str  # "rrf", "rerank"


class HybridRetriever:
    """Hybrid retriever combining BM25 + dense via RRF, with optional reranker."""

    def __init__(self, use_reranker: bool = True):
        self.chunks = pd.read_pickle(INDEX_DIR / "chunks.pkl")
        self.faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["bm25"]
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL) if use_reranker else None

    # ------------- single retrievers -------------
    def _dense_top(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.faiss_index.search(q, k)
        return list(zip(idxs[0].tolist(), scores[0].tolist()))

    def _bm25_top(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        from src.index import tokenize  # reuse tokenizer
        scores = self.bm25.get_scores(tokenize(query))
        top_idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idxs]

    # ------------- fusion -------------
    @staticmethod
    def _rrf(ranked_lists: list[list[int]], k_rrf: int = 60) -> dict[int, float]:
        """Reciprocal Rank Fusion (Cormack et al. 2009)."""
        out: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                out[idx] = out.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
        return out

    # ------------- public -------------
    def retrieve(self, query: str, candidates: int = 20, final_k: int = 5) -> list[Hit]:
        dense = self._dense_top(query, candidates)
        sparse = self._bm25_top(query, candidates)

        fused = self._rrf([
            [idx for idx, _ in dense],
            [idx for idx, _ in sparse],
        ])
        # Top by RRF score
        ranked_idxs = sorted(fused, key=fused.get, reverse=True)[:candidates]

        candidates_hits: list[Hit] = []
        for idx in ranked_idxs:
            row = self.chunks.iloc[idx]
            candidates_hits.append(Hit(
                doc_id=row["doc_id"],
                chunk_id=row["chunk_id"],
                title=row["title"],
                text=row["text"],
                score=fused[idx],
                source="rrf",
            ))

        # Optional reranker
        if self.reranker is not None and candidates_hits:
            pairs = [[query, h.text] for h in candidates_hits]
            ce_scores = self.reranker.predict(pairs)
            for h, s in zip(candidates_hits, ce_scores):
                h.score = float(s)
                h.source = "rerank"
            candidates_hits.sort(key=lambda h: h.score, reverse=True)

        return candidates_hits[:final_k]


def main() -> None:
    """CLI smoke test."""
    import sys
    r = HybridRetriever(use_reranker=True)
    q = " ".join(sys.argv[1:]) or "How long do I have to return an item?"
    print(f"\nQuery: {q}\n")
    for i, h in enumerate(r.retrieve(q), 1):
        print(f"#{i} [{h.chunk_id}] score={h.score:.4f} ({h.source})")
        print(f"  {h.text[:200]}...")
        print()


if __name__ == "__main__":
    main()
