from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Incoming


@dataclass(frozen=True)
class Policy:
    """Who may use the bot. Deny-by-default; set open=true to accept anyone."""

    allow_chats: frozenset[str] = frozenset()
    allow_senders: frozenset[str] = frozenset()
    max_duration_s: float | None = None
    open: bool = False

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Policy:
        return cls(
            allow_chats=frozenset(map(str, options.get("allow_chats", []))),
            allow_senders=frozenset(map(str, options.get("allow_senders", []))),
            max_duration_s=options.get("max_duration_s"),
            open=bool(options.get("open", False)),
        )

    def check(self, msg: Incoming) -> str | None:
        """Return a rejection reason, or None if the message is allowed."""
        if not self.open and msg.chat_id not in self.allow_chats and msg.sender_id not in self.allow_senders:
            return "not allowlisted"
        duration_s = getattr(msg, "duration_s", None)
        if self.max_duration_s is not None and duration_s and duration_s > self.max_duration_s:
            return f"longer than {self.max_duration_s}s"
        return None
