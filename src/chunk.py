"""Markdown-aware semantic chunking with sentence-boundary fallback.

Upgrade from baseline 800/150 fixed-character splitting:
- Split on paragraph boundaries first
- Pack paragraphs into target ~800-char windows
- Never cut mid-sentence
- 1-2 sentence overlap between adjacent chunks
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TARGET_SIZE = 800
MAX_SIZE = 1200
OVERLAP_SENTS = 2

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def chunk_text(text: str, target: int = TARGET_SIZE, max_size: int = MAX_SIZE) -> list[str]:
    """Pack paragraphs greedily up to target size, split oversized ones by sentence."""
    chunks: list[str] = []
    buffer = ""
    prev_sents: list[str] = []

    paras = split_paragraphs(text)
    for para in paras:
        if len(para) > max_size:
            # paragraph too long: split by sentence and recurse
            sents = split_sentences(para)
            current = ""
            for sent in sents:
                if len(current) + len(sent) + 1 <= target:
                    current = f"{current} {sent}".strip()
                else:
                    if current:
                        chunks.append(current)
                        prev_sents = split_sentences(current)[-OVERLAP_SENTS:]
                    current = (" ".join(prev_sents) + " " + sent).strip() if prev_sents else sent
            if current:
                chunks.append(current)
                prev_sents = split_sentences(current)[-OVERLAP_SENTS:]
            buffer = ""
            continue

        if len(buffer) + len(para) + 1 <= target:
            buffer = f"{buffer}\n\n{para}".strip()
        else:
            if buffer:
                chunks.append(buffer)
                prev_sents = split_sentences(buffer)[-OVERLAP_SENTS:]
            buffer = (" ".join(prev_sents) + "\n\n" + para).strip() if prev_sents else para

    if buffer:
        chunks.append(buffer)

    return chunks


def main() -> None:
    manifest = pd.read_csv(ROOT / "data" / "docs_manifest.csv")
    rows = []
    for _, row in manifest.iterrows():
        doc_id = row["doc_id"]
        text = Path(row["text_path"]).read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            rows.append({
                "doc_id": doc_id,
                "title": row["title"],
                "chunk_id": f"{doc_id}_{i:04d}",
                "text": ch,
                "n_chars": len(ch),
            })
    chunk_df = pd.DataFrame(rows)
    out = ROOT / "data" / "chunks.pkl"
    chunk_df.to_pickle(out)
    print(f"Wrote {len(chunk_df)} chunks → {out}")
    print(chunk_df.groupby("doc_id").size().to_string())


if __name__ == "__main__":
    main()
