"""Placement wording for add prompts: where the object goes and what it rests on."""

from __future__ import annotations

GUIDE = (
    "Match the scene's lighting, scale and perspective, with a natural contact "
    "shadow where it touches a surface."
)


def placement(anchor: str | None) -> str:
    """`' on the bench'` for an anchor phrase, empty without one."""
    anchor = (anchor or "").strip().rstrip(".")
    return f" {anchor}" if anchor else ""

