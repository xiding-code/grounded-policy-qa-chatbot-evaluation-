"""HTML → clean text. Strips nav, scripts, repeated boilerplate."""
from __future__ import annotations
from pathlib import Path
import re
from bs4 import BeautifulSoup
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TEXT = ROOT / "data" / "text"
TEXT.mkdir(parents=True, exist_ok=True)

BOILERPLATE_PATTERNS = [
    r"^\s*Skip to main content\s*$",
    r"^\s*Returns & Refunds\s*$",
    r"^\s*Help & Customer Service\s*$",
    r"^\s*Was this information helpful\?.*$",
    r"^\s*Yes\s*$",
    r"^\s*No\s*$",
    r"^\s*Thank you for your feedback\..*$",
    r"^\s*Conditions of Use.*$",
    r"^\s*Privacy Notice.*$",
    r"^\s*Cookies Notice.*$",
    r"^\s*Interest-Based Ads Notice.*$",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def html_to_text(html: str) -> str:
    """Extract main help-content text, drop nav + script + repeated lines."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        t.decompose()

    # Prefer the help-content container if present; else fall back to body
    main = (
        soup.find(id="help-content")
        or soup.find("div", {"class": "help-content"})
        or soup.find("main")
        or soup.find("body")
    )
    raw_text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    lines = [ln.strip() for ln in raw_text.splitlines()]
    # Drop boilerplate + empty + duplicates while preserving order
    seen = set()
    cleaned = []
    for ln in lines:
        if not ln or BOILERPLATE_RE.match(ln):
            continue
        if ln in seen:
            continue
        seen.add(ln)
        cleaned.append(ln)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    manifest = pd.read_csv(ROOT / "data" / "docs_manifest.csv")
    rows = []
    for _, row in manifest.iterrows():
        doc_id = row["doc_id"]
        html = Path(row["raw_path"]).read_text(encoding="utf-8")
        text = html_to_text(html)
        out = TEXT / f"{doc_id}.txt"
        out.write_text(text, encoding="utf-8")
        rows.append({**row.to_dict(), "text_path": str(out), "text_chars": len(text)})
        print(f"[cleaned] {doc_id}: {len(text)} chars")
    pd.DataFrame(rows).to_csv(ROOT / "data" / "docs_manifest.csv", index=False)


if __name__ == "__main__":
    main()
