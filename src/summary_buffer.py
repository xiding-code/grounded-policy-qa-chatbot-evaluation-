"""L2 Conversation Summary Buffer — compress older turns when budget exceeded.

When the rolling buffer exceeds `keep_raw_turns`, older turns are
summarized into a single condensed system note via a cheap LLM call.
The summary persists; only the most recent N turns stay verbatim.

Why this matters:
  - Token budget bounded even for 20+ turn conversations
  - Long-term context preserved (key facts, decisions, entities)
  - Cost stays roughly flat per turn instead of growing O(n)

Falls back gracefully if no LLM is configured (just drops oldest turns).
"""
from __future__ import annotations
from dataclasses import dataclass

from src.llm import call_llm, is_configured


SUMMARIZER_SYSTEM = """You are a conversation summarizer for a support chatbot.

Given a list of past user-assistant turns, write a single concise summary paragraph (under 80 words) capturing:
  - the user's overall goal
  - any specific entities mentioned (order IDs, products, dates)
  - key decisions already reached (approved? denied? escalated?)
  - open questions still pending

Output the summary text only. No preamble. No bullet points. No quotes."""


@dataclass
class SummarizedBuffer:
    summary: str           # condensed prior context
    raw_turns: list[dict]  # most recent N turns verbatim
    n_compressed: int      # how many turns were folded into the summary

    def messages(self) -> list[dict]:
        """Return messages in OpenAI/Anthropic format with summary prepended."""
        out: list[dict] = []
        if self.summary:
            # Inject summary as a fake assistant context note so the LLM can read it
            out.append({"role": "user", "content": f"[Conversation context so far]: {self.summary}"})
            out.append({"role": "assistant", "content": "Understood. Continuing the conversation."})
        out.extend(self.raw_turns)
        return out


def compress(
    all_turns: list[dict],
    keep_raw_turns: int = 4,
    prior_summary: str | None = None,
) -> SummarizedBuffer:
    """If we have more than `keep_raw_turns` turns, summarize the older ones.

    `all_turns`: full conversation in message format
    `keep_raw_turns`: number of MOST RECENT turns (user+assistant pairs) to keep verbatim
    `prior_summary`: existing summary from a previous compression (folded in)
    """
    keep_msgs = keep_raw_turns * 2  # 2 messages per turn

    if len(all_turns) <= keep_msgs and not prior_summary:
        return SummarizedBuffer(summary="", raw_turns=list(all_turns), n_compressed=0)

    to_compress = all_turns[:-keep_msgs] if len(all_turns) > keep_msgs else []
    raw = all_turns[-keep_msgs:] if len(all_turns) > keep_msgs else list(all_turns)
    n_to_compress = len(to_compress) // 2

    if not to_compress and not prior_summary:
        return SummarizedBuffer(summary="", raw_turns=raw, n_compressed=0)

    ok, _ = is_configured()
    if not ok:
        # No LLM - just drop the older turns, keep a minimal note
        note = f"[{n_to_compress} earlier turns dropped — LLM not available for summarization]"
        if prior_summary:
            note = f"{prior_summary} {note}"
        return SummarizedBuffer(summary=note, raw_turns=raw, n_compressed=n_to_compress)

    # Build the input for the summarizer
    transcript = ""
    if prior_summary:
        transcript += f"Earlier summary: {prior_summary}\n\n"
    transcript += "New turns to fold in:\n"
    for m in to_compress:
        role = m.get("role", "?").upper()
        content = m.get("content", "")[:300]  # truncate to keep summarizer input bounded
        transcript += f"{role}: {content}\n"

    try:
        summary = call_llm(SUMMARIZER_SYSTEM, transcript).strip()
    except Exception:
        summary = f"[Summary unavailable. {n_to_compress} earlier turns dropped.]"

    return SummarizedBuffer(
        summary=summary,
        raw_turns=raw,
        n_compressed=n_to_compress + (1 if prior_summary else 0),
    )
