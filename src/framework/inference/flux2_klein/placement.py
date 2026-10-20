"""Placement wording for add prompts: where the object goes and what it rests on."""

from __future__ import annotations

import os
from io import BytesIO

from PIL import Image, ImageDraw

ANCHOR_SYSTEM_PROMPT = (
    "You describe where a new object will be placed in a photo. The red rectangle "
    "marks the target region. Reply with ONE short placement phrase of at most "
    "8 words: a preposition plus the existing object or surface at that region "
    "(for example 'on the blue cooler' or 'on the wooden floor'). Mention the "
    "region's own surface or object only, never the new object and never other "
    "objects or animals nearby. No other text."
)
REPLACE_NOTE = (
    " The region currently holds an object that is being replaced: describe the "
    "ground or surface it stands on or hangs from, not the object itself."
)
GUIDE = (
    "Match the scene's lighting, scale and perspective, with a natural contact "
    "shadow where it touches a surface."
)


def placement(anchor: str | None) -> str:
    """`' on the bench'` for an anchor phrase, empty without one."""
    anchor = (anchor or "").strip().rstrip(".")
    return f" {anchor}" if anchor else ""


def describe_anchor(
    image: Image.Image,
    box: tuple[int, int, int, int],
    prompt: str,
    *,
    client=None,
    model: str | None = None,
    replacing: bool = False,
) -> str | None:
    """Ask Gemini what the region sits on; ``None`` when unavailable or it fails."""
    try:
        from google.genai import types

        if client is None:
            from google import genai

            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                return None
            client = genai.Client(api_key=key)
        raw = model or os.getenv("ANNOTATE_LLM", "google/gemini-2.5-flash-lite")
        model = raw.split("/", 1)[1] if raw.startswith("google/") else raw
        marked = image.convert("RGB").copy()
        ImageDraw.Draw(marked).rectangle(
            box, outline=(255, 0, 0), width=max(2, image.width // 200)
        )
        buffer = BytesIO()
        marked.save(buffer, format="PNG")
        response = client.models.generate_content(
            model=model,
            contents=[
                f"New object to add: {prompt}",
                types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                system_instruction=ANCHOR_SYSTEM_PROMPT
                + (REPLACE_NOTE if replacing else ""),
                temperature=0,
            ),
        )
        text = " ".join((response.text or "").split()).strip(" .\"'")
    except Exception:
        return None
    return text if 0 < len(text.split()) <= 12 else None
