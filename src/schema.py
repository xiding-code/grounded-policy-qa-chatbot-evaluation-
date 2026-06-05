"""Structured output schema + safety guard.

Schema (4 decision labels):
  APPROVE | DENY | ESCALATE | ABSTAIN

Safety guard: APPROVE / DENY without CITATIONS → force ABSTAIN.
Preserves the citation-grounded decision discipline from the baseline.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import re

ALLOWED_DECISIONS = {"APPROVE", "DENY", "ESCALATE", "ABSTAIN"}

SYSTEM_PROMPT = """You are a retail customer-support assistant for Amazon returns and refunds.

Use ONLY the EVIDENCE passages provided. Do not rely on outside knowledge.

Output MUST follow this exact format:

DECISION: <APPROVE|DENY|ESCALATE|ABSTAIN>
ANSWER: <one or two short sentences>
REQUIRED_INFO: <NONE or what extra info is needed>
CITATIONS: <NONE or list of chunk IDs separated by commas>

Rules:
- APPROVE if evidence clearly supports the user's request.
- DENY if evidence clearly contradicts the user's request.
- ESCALATE if the user needs account-specific action (order lookup, disputes, chargebacks, fraud).
- ABSTAIN if EVIDENCE is insufficient, unclear, or missing.
- ALWAYS cite the chunk IDs you used. If you cannot cite, output ABSTAIN.
- Never invent facts not present in the EVIDENCE.
"""


@dataclass
class Decision:
    decision: str
    answer: str
    required_info: str
    citations: str
    raw: str
    guarded: bool = False  # True if safety guard overrode the decision

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def parse_structured(raw: str) -> Decision:
    """Parse LLM raw output into a Decision. Apply safety guard."""
    out = {"DECISION": "ABSTAIN", "ANSWER": "", "REQUIRED_INFO": "NONE", "CITATIONS": "NONE"}
    for line in raw.splitlines():
        line = line.strip()
        for key in out.keys():
            if line.upper().startswith(f"{key}:"):
                out[key] = line.split(":", 1)[1].strip()

    decision = out["DECISION"].upper()
    if decision not in ALLOWED_DECISIONS:
        decision = "ABSTAIN"
    out["DECISION"] = decision

    guarded = False
    if decision in {"APPROVE", "DENY"}:
        cites = out["CITATIONS"].strip().upper()
        if cites in {"", "NONE"}:
            out["DECISION"] = "ABSTAIN"
            out["ANSWER"] = "Insufficient cited evidence; abstaining."
            guarded = True

    return Decision(
        decision=out["DECISION"],
        answer=out["ANSWER"],
        required_info=out["REQUIRED_INFO"],
        citations=out["CITATIONS"],
        raw=raw,
        guarded=guarded,
    )


def build_user_prompt(question: str, evidence_chunks: list[tuple[str, str]]) -> str:
    """evidence_chunks = [(chunk_id, text), ...]"""
    lines = ["EVIDENCE:"]
    for chunk_id, text in evidence_chunks:
        lines.append(f"[{chunk_id}] {text}")
    lines.append("")
    lines.append(f"QUESTION: {question}")
    return "\n".join(lines)


# Rule-based fallback for demo when no LLM is wired up
def heuristic_decide(question: str, hits: list) -> Decision:
    """Deterministic stub: if we have hits, ABSTAIN + cite top-3.
    Documents the contract without requiring an LLM key."""
    q_lower = question.lower()
    citations = ", ".join(h.chunk_id for h in hits[:3]) if hits else "NONE"

    if not hits:
        raw = "DECISION: ABSTAIN\nANSWER: No relevant policy chunks retrieved.\nREQUIRED_INFO: NONE\nCITATIONS: NONE"
        return parse_structured(raw)

    # Cheap intent detection for demo only
    if any(w in q_lower for w in ["my order", "my refund", "my account", "dispute", "chargeback"]):
        raw = (
            f"DECISION: ESCALATE\n"
            f"ANSWER: This requires account-specific action by a human agent.\n"
            f"REQUIRED_INFO: order ID, payment method, account email\n"
            f"CITATIONS: {citations}"
        )
        return parse_structured(raw)

    # Default: abstain with citations so an LLM can take over in production
    raw = (
        f"DECISION: ABSTAIN\n"
        f"ANSWER: Retrieved evidence available; LLM generation not wired in this build.\n"
        f"REQUIRED_INFO: NONE\n"
        f"CITATIONS: {citations}"
    )
    return parse_structured(raw)
