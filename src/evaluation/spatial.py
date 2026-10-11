from __future__ import annotations

import re


def needs_spatial_anchor_box(instruction: str) -> bool:
    """Return whether an instruction places an object relative to an anchor."""
    return bool(
        re.search(r"\b(?:add|place|put|insert|position)\b", instruction, re.IGNORECASE)
        and re.search(
            r"\b(?:beside|next to|near|behind|in front of|to the (?:left|right) of|"
            r"(?:front|rear|back) of)\b",
            instruction,
            re.IGNORECASE,
        )
    )
