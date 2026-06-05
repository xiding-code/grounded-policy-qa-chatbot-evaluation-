"""L3 Entity Tracker — extract and persist structured slots from conversation.

Lightweight rule-based + LLM-assisted slot filler. Tracks:
  - order_id        (e.g. "112-3456789-1234567")
  - product         (e.g. "MacBook Pro", "iPhone", "laptop")
  - return_reason   (damaged | defective | wrong-item | changed-mind | late | other)
  - intent          (return | refund | track | exchange | escalate | inquiry)

Why this matters:
  - Multi-turn coreference: "Did the laptop ship?" -> we know which laptop
  - Personalization: address user by their tracked entities
  - Escalation: when ESCALATE fires, the structured slots are ready
    to hand to a human agent (zero hand-off friction)

Pattern: rule-based regex catches obvious cases (order IDs); a tiny
LLM call fills the rest. Falls back to rule-only if no LLM key.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import re

from src.llm import call_llm, is_configured, LLMNotConfigured


ORDER_ID_RE = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")  # Amazon order format

PRODUCT_KEYWORDS = [
    "laptop", "macbook", "iphone", "ipad", "kindle", "echo", "phone",
    "tablet", "monitor", "tv", "television", "camera", "headphones",
    "watch", "speaker", "router", "printer", "keyboard", "mouse",
    "book", "shoes", "shirt", "dress", "jacket", "bag", "backpack",
    "appliance", "furniture", "chair", "desk", "lamp", "mattress",
]

REASON_KEYWORDS = {
    "damaged": ["damaged", "broken", "cracked", "shattered", "dented", "torn"],
    "defective": ["defective", "doesn't work", "doesnt work", "not working", "faulty", "malfunction"],
    "wrong-item": ["wrong item", "wrong product", "different item", "not what i ordered", "not what I ordered"],
    "late": ["late", "never arrived", "still hasn't", "delayed", "missing"],
    "changed-mind": ["changed my mind", "don't want", "dont want", "no longer need", "doesn't fit", "doesnt fit"],
}

INTENT_KEYWORDS = {
    "return": ["return", "send back"],
    "refund": ["refund", "money back", "reimburse"],
    "track": ["track", "where is", "shipped", "delivery status"],
    "exchange": ["exchange", "replace", "swap"],
    "escalate": ["complaint", "speak to manager", "human", "agent", "supervisor"],
}


@dataclass
class Entities:
    order_id: str | None = None
    product: str | None = None
    return_reason: str | None = None  # damaged | defective | wrong-item | changed-mind | late | other
    intent: str | None = None         # return | refund | track | exchange | escalate | inquiry
    updated_keys: list[str] = field(default_factory=list)  # keys touched this turn

    def merge(self, new: "Entities") -> list[str]:
        """Merge non-None values from `new` into self. Returns list of updated keys."""
        updated: list[str] = []
        for key in ("order_id", "product", "return_reason", "intent"):
            new_val = getattr(new, key)
            if new_val and new_val != getattr(self, key):
                setattr(self, key, new_val)
                updated.append(key)
        self.updated_keys = updated
        return updated

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("updated_keys", None)
        d["any_set"] = any(d.values())
        return d

    def context_lines(self) -> list[str]:
        """Format for injection into system prompt."""
        lines = []
        if self.order_id:    lines.append(f"- Tracked order_id: {self.order_id}")
        if self.product:     lines.append(f"- Tracked product: {self.product}")
        if self.return_reason: lines.append(f"- Return reason: {self.return_reason}")
        if self.intent:      lines.append(f"- User intent: {self.intent}")
        return lines


def _rule_extract(question: str) -> Entities:
    q_lower = question.lower()
    out = Entities()

    m = ORDER_ID_RE.search(question)
    if m:
        out.order_id = m.group(0)

    for kw in PRODUCT_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", q_lower):
            out.product = kw
            break

    for reason, keywords in REASON_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            out.return_reason = reason
            break

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            out.intent = intent
            break

    return out


_LLM_SYSTEM = """Extract entity slots from a customer support message.
Return a single JSON object with these keys (use null when unknown):
  - order_id:      Amazon order ID format "XXX-XXXXXXX-XXXXXXX", else null
  - product:       short product noun (e.g. "laptop", "iphone", "kindle"), else null
  - return_reason: one of "damaged", "defective", "wrong-item", "changed-mind", "late", "other", null
  - intent:        one of "return", "refund", "track", "exchange", "escalate", "inquiry", null

Output JSON only. No prose, no markdown fences."""


def _llm_extract(question: str) -> Entities:
    try:
        raw = call_llm(_LLM_SYSTEM, f"Message: {question}\n\nJSON:")
        # Strip any fence if model added one
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        return Entities(
            order_id=data.get("order_id") or None,
            product=data.get("product") or None,
            return_reason=data.get("return_reason") or None,
            intent=data.get("intent") or None,
        )
    except (LLMNotConfigured, json.JSONDecodeError, Exception):
        return Entities()


def extract(question: str, use_llm: bool = True) -> Entities:
    """Extract entities from a question. Combines rule-based + optional LLM.

    Rule pass runs first (cheap, high precision on order_id).
    If LLM is configured and rules missed product/intent, LLM fills gaps.
    """
    rule = _rule_extract(question)
    if not use_llm or not is_configured()[0]:
        return rule

    # Skip LLM call if rules already filled all key slots
    if rule.order_id and rule.product and rule.intent:
        return rule

    llm = _llm_extract(question)
    # Rules win on order_id (regex is exact); LLM wins on softer slots if missing
    merged = Entities(
        order_id=rule.order_id or llm.order_id,
        product=rule.product or llm.product,
        return_reason=rule.return_reason or llm.return_reason,
        intent=rule.intent or llm.intent,
    )
    return merged


class EntityStore:
    """Per-session entity state. Lives next to SessionManager."""

    def __init__(self):
        self._store: dict[str, Entities] = {}

    def get(self, session_id: str) -> Entities:
        if session_id not in self._store:
            self._store[session_id] = Entities()
        return self._store[session_id]

    def update(self, session_id: str, new_entities: Entities) -> list[str]:
        current = self.get(session_id)
        return current.merge(new_entities)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
