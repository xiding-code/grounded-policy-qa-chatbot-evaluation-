"""LLM provider abstraction. Reads .env for provider + key + model.

Supports Anthropic Claude and OpenAI GPT, with optional conversation
history (multi-turn). Adding a new provider = adding a new branch in `call_llm`.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (parent of src/)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))


class LLMNotConfigured(RuntimeError):
    pass


def _key_looks_real(value: str | None) -> bool:
    if not value:
        return False
    if "PASTE-YOUR" in value.upper():
        return False
    if len(value) < 20:
        return False
    return True


def call_anthropic(
    system_prompt: str,
    user_prompt: str,
    history: list[dict] | None = None,
) -> str:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY")
    if not _key_looks_real(key):
        raise LLMNotConfigured("ANTHROPIC_API_KEY is missing or still a placeholder in .env")
    client = anthropic.Anthropic(api_key=key)

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=messages,
    )
    parts = []
    for block in resp.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts).strip()


def call_openai(
    system_prompt: str,
    user_prompt: str,
    history: list[dict] | None = None,
) -> str:
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not _key_looks_real(key):
        raise LLMNotConfigured("OPENAI_API_KEY is missing or still a placeholder in .env")
    client = OpenAI(api_key=key)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    history: list[dict] | None = None,
    provider: str | None = None,
) -> str:
    """Dispatch based on .env LLM_PROVIDER, overridable per-call.

    history: optional [{"role": "user|assistant", "content": str}, ...] for multi-turn.
    """
    p = (provider or LLM_PROVIDER).lower().strip()
    if p == "anthropic":
        return call_anthropic(system_prompt, user_prompt, history=history)
    if p == "openai":
        return call_openai(system_prompt, user_prompt, history=history)
    raise LLMNotConfigured(f"Unknown LLM_PROVIDER: {p!r}. Use 'anthropic' or 'openai'.")


def is_configured() -> tuple[bool, str]:
    """Quick check whether the currently selected provider has a real key."""
    if LLM_PROVIDER == "anthropic":
        return _key_looks_real(os.getenv("ANTHROPIC_API_KEY")), f"anthropic / {ANTHROPIC_MODEL}"
    if LLM_PROVIDER == "openai":
        return _key_looks_real(os.getenv("OPENAI_API_KEY")), f"openai / {OPENAI_MODEL}"
    return False, f"unknown provider {LLM_PROVIDER!r}"
