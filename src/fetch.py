"""Fetch Amazon returns/refunds help pages with browser headers (bypasses 403)."""
from __future__ import annotations
import time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Curated corpus from original BA840 project + supplements.
DOCS = [
    {"doc_id": "A1", "title": "Return Items You Ordered",
     "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=G6E3B2E8QPHQ88KF"},
    {"doc_id": "A3", "title": "Returns to Third-Party Sellers",
     "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=G38BHJQ25PNCLUBU"},
    {"doc_id": "A4", "title": "Amazon Return Policy",
     "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7"},
    {"doc_id": "A6", "title": "Track Your Return",
     "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=GNF2KMBB2JD4VXV8"},
    {"doc_id": "A7", "title": "Return Shipping Cost",
     "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=GFLBEJCLHMVFEPA8"},
    {"doc_id": "A9", "title": "International Returns",
     "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=GP8L6BMXBTJHKUJW"},
]


def fetch_one(doc_id: str, url: str) -> Path:
    """Idempotent: skip if cached."""
    out = RAW / f"{doc_id}.html"
    if out.exists() and out.stat().st_size > 1000:
        print(f"[cache] {doc_id}: {out.stat().st_size} bytes")
        return out
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    out.write_text(r.text, encoding="utf-8")
    print(f"[fetched] {doc_id}: {len(r.text)} bytes")
    time.sleep(0.5)  # be polite
    return out


def main() -> None:
    rows = []
    for d in DOCS:
        path = fetch_one(d["doc_id"], d["url"])
        rows.append({**d, "raw_path": str(path), "raw_bytes": path.stat().st_size})
    manifest = pd.DataFrame(rows)
    manifest_path = ROOT / "data" / "docs_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest: {manifest_path} ({len(manifest)} docs)")


if __name__ == "__main__":
    main()
