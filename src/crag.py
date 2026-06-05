"""CRAG — Corrective Retrieval-Augmented Generation.

Pattern (Yan et al. 2024, simplified):
  After retrieval, ask the LLM to score the quality:
    - HIGH: retrieved chunks directly answer the question
    - LOW:  retrieved chunks are off-topic, contradictory, or insufficient

  HIGH -> proceed normally
  LOW  -> trigger web fallback (DuckDuckGo Instant Answer API for demo)
          and append fallback evidence to the LLM context

Why this matters:
  - Catches the case where the corpus has a gap (e.g. a brand-new policy
    not yet ingested) and provides graceful degradation
  - Adds an auditable confidence signal to every answer

For the demo we use DuckDuckGo because it requires no API key. In
production swap for Tavily / Brave / Serper / Bing depending on
your stack.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import re

import requests

from src.llm import call_llm, is_configured


JUDGE_SYSTEM = """You are a retrieval quality judge.

Given a user question and a list of retrieved policy chunks, decide if the chunks contain enough information to answer the question accurately.

Output a single JSON object:
{
  "quality": "HIGH" | "LOW",
  "reason": "one short sentence explaining the verdict"
}

Output JSON only. No prose, no markdown fences."""


@dataclass
class QualityVerdict:
    quality: str         # "HIGH" or "LOW"
    reason: str
    used_llm: bool
    fallback: list[dict] = field(default_factory=list)  # web fallback evidence if LOW


def judge(question: str, evidence_text: str) -> QualityVerdict:
    """LLM judges whether retrieved evidence is sufficient."""
    ok, _ = is_configured()
    if not ok:
        return QualityVerdict(quality="HIGH", reason="no-llm-defaults-high", used_llm=False)

    user = f"Question: {question}\n\nRetrieved evidence:\n{evidence_text}\n\nVerdict:"
    try:
        raw = call_llm(JUDGE_SYSTEM, user).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        q = data.get("quality", "HIGH").upper()
        if q not in {"HIGH", "LOW"}:
            q = "HIGH"
        return QualityVerdict(
            quality=q,
            reason=str(data.get("reason", ""))[:200],
            used_llm=True,
        )
    except (json.JSONDecodeError, Exception) as e:
        return QualityVerdict(quality="HIGH", reason=f"judge-error-defaults-high: {type(e).__name__}", used_llm=False)


def duckduckgo_fallback(query: str, max_results: int = 3) -> list[dict]:
    """DuckDuckGo Instant Answer API. Returns a list of fallback chunks.

    Each chunk has the same shape as our regular evidence so the UI
    can render it uniformly.
    """
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=8,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out = []

        # AbstractText is the headline summary
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            out.append({
                "chunk_id": "WEB_abstract",
                "doc_id": "WEB",
                "title": data.get("Heading") or "Web fallback",
                "score": 1.0,
                "source": "web",
                "text_preview": abstract[:300] + ("..." if len(abstract) > 300 else ""),
                "_text_full": abstract,
            })

        # RelatedTopics are bullets
        for i, topic in enumerate((data.get("RelatedTopics") or [])[:max_results]):
            if isinstance(topic, dict) and topic.get("Text"):
                text = topic["Text"].strip()
                out.append({
                    "chunk_id": f"WEB_topic_{i+1}",
                    "doc_id": "WEB",
                    "title": (topic.get("FirstURL") or "Web result").rsplit("/", 1)[-1].replace("_", " "),
                    "score": 0.8 - i * 0.05,
                    "source": "web",
                    "text_preview": text[:300] + ("..." if len(text) > 300 else ""),
                    "_text_full": text,
                })

        return out
    except Exception:
        return []


def maybe_correct(
    question: str,
    evidence_chunks: list[dict],
    enable_web_fallback: bool = True,
) -> QualityVerdict:
    """Full CRAG step: judge quality, optionally fetch web fallback if LOW."""
    evidence_text = "\n".join(
        f"[{c.get('chunk_id', '?')}] {c.get('text_preview') or c.get('text', '')}"
        for c in evidence_chunks[:5]
    )
    verdict = judge(question, evidence_text)

    if verdict.quality == "LOW" and enable_web_fallback:
        verdict.fallback = duckduckgo_fallback(f"Amazon returns refunds {question}")

    return verdict
