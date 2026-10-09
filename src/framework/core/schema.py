from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Message:
    """User-facing message produced by ``send_message``."""

    text: str
    objects: list[Any]


@dataclass(slots=True)
class TurnResult:
    """Result of one user turn in a persistent conversation."""

    message: Message
    raw: Any

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(getattr(self.raw, "steps", []) or [])


RunResult = TurnResult
