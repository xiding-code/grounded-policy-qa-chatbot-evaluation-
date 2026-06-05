"""Conversation memory layer — L1 rolling buffer.

Session-keyed in-memory store. Each session keeps the last N turns
(user + assistant) for multi-turn context injection.

Upgrades over baseline (single-turn stateless):
- User can follow up with "what about shipping for it" and the bot
  retains the prior product reference.
- Token cost stays bounded by max_turns cap.

Roadmap (not yet shipped):
- L2 ConversationSummaryBufferMemory: summarize older turns with cheap LLM
- L3 Entity Memory: track order_id / product / step_in_flow as structured slots
- Persistence: swap dict-backed store for Redis (hot session) + Postgres (profile)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import time


Role = Literal["user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str
    ts: float = field(default_factory=time.time)


class ConversationBuffer:
    """Rolling window over conversation turns.

    Caps at `max_turns * 2` messages (user + assistant per turn).
    Returns history in the format LLM SDKs expect.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add(self, role: Role, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        # Cap at max_turns * 2 (each turn = 1 user + 1 assistant message)
        cap = self.max_turns * 2
        if len(self._turns) > cap:
            self._turns = self._turns[-cap:]

    def messages(self) -> list[dict]:
        """Return turns in OpenAI/Anthropic message format."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def n_turns(self) -> int:
        """Number of complete user+assistant turns."""
        return len(self._turns) // 2

    def clear(self) -> None:
        self._turns = []


class SessionManager:
    """In-memory session registry. Swap for Redis in production."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._sessions: dict[str, ConversationBuffer] = {}

    def get(self, session_id: str) -> ConversationBuffer:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationBuffer(self.max_turns)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def n_sessions(self) -> int:
        return len(self._sessions)
