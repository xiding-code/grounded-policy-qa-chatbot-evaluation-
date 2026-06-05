"""Self-RAG — LLM-based router that decides whether to retrieve.

Pattern (simplified from Asai et al. 2023):
  Before running retrieval, ask the LLM:
    "Does answering this need policy lookup? Yes or No."
  If No, skip retrieval. The downstream LLM answers directly (or refuses
  politely if the question is out-of-scope for a returns/refunds bot).

Why this matters:
  - Casual greetings ("hi", "thanks") don't need 200ms of retrieval
  - Out-of-scope questions ("what's the weather?") get a polite redirect
  - Cuts latency on ~20-30 percent of real-world support traffic

Fallback: if LLM not configured, route everything through retrieval (safe).
"""
from __future__ import annotations
from dataclasses import dataclass

from src.llm import call_llm, is_configured, LLMNotConfigured


SELF_RAG_SYSTEM = """You are a routing classifier for an Amazon returns/refunds support chatbot.

Decide whether the user's latest message requires looking up Amazon's returns/refunds policy documentation, or whether it can be answered directly (greetings, thanks, off-topic clarifications, scope refusal).

Output exactly one word:
  RETRIEVE  - the message needs policy lookup (return windows, eligibility, shipping, refund methods, third-party rules, etc.)
  SKIP      - greeting, thank-you, chit-chat, clearly out of scope, or a pure follow-up clarification on something already answered

No other output. No punctuation. No explanation."""


@dataclass
class RouteDecision:
    route: str          # "RETRIEVE" or "SKIP"
    reason: str         # short tag for UI
    used_llm: bool      # true if real LLM call was made


def route(question: str, force_retrieve: bool = False) -> RouteDecision:
    """Classify whether to run retrieval."""
    if force_retrieve:
        return RouteDecision(route="RETRIEVE", reason="forced", used_llm=False)

    # Cheap heuristic short-circuit for obvious cases
    q_clean = question.strip().lower()
    if len(q_clean) <= 6 and q_clean in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no"}:
        return RouteDecision(route="SKIP", reason="greeting", used_llm=False)

    ok, _ = is_configured()
    if not ok:
        # No LLM available - route everything through retrieval (safe default)
        return RouteDecision(route="RETRIEVE", reason="default-no-llm", used_llm=False)

    try:
        raw = call_llm(SELF_RAG_SYSTEM, question)
        verdict = raw.strip().upper().split()[0] if raw.strip() else "RETRIEVE"
        if verdict.startswith("SKIP"):
            return RouteDecision(route="SKIP", reason="llm-classified", used_llm=True)
        return RouteDecision(route="RETRIEVE", reason="llm-classified", used_llm=True)
    except (LLMNotConfigured, Exception):
        return RouteDecision(route="RETRIEVE", reason="llm-error-default", used_llm=False)


SKIP_RESPONSE_SYSTEM = """You are an Amazon returns/refunds support assistant. The user's last message did not require a policy lookup.

If it's a greeting or thank-you, respond warmly in one sentence.
If it's out of scope (not about Amazon returns/refunds), gently redirect to your scope in one sentence.
Output ONLY the response text. No JSON, no labels, no preamble."""


def answer_skip(question: str) -> str:
    """Generate a direct response when retrieval was skipped."""
    ok, _ = is_configured()
    if not ok:
        return "I'm a returns and refunds policy assistant. How can I help you today?"
    try:
        return call_llm(SKIP_RESPONSE_SYSTEM, question)
    except Exception:
        return "I'm a returns and refunds policy assistant. How can I help you today?"
