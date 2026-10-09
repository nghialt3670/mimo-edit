from __future__ import annotations

from PIL import Image, ImageDraw

from framework.utils.image import validate_box
from framework.inference import ModelCache, edit_masked_object
from ..registry import register_tool


@register_tool(
    "generate_object",
    "Generate or transform content inside one selected region using FLUX.2 Klein. Provide one mask or one box and describe the desired result in prompt. For phrase-controlled placement, provide a box and phrase; do not provide a mask in this mode.",
    {
        "image": {"type": "image", "description": "Source image."},
        "prompt": {"type": "string", "description": "Describe the requested content or transformation inside the selected region."},
        "mask": {
            "type": "image",
            "description": "Optional Pillow mask; white pixels select the one region to generate or transform. Pass null when using a box.",
            "nullable": True,
        },
        "box": {
            "type": "array",
            "description": "Optional pixel box [x1, y1, x2, y2] for the one region. Pair it with phrase for phrase-guided placement.",
            "nullable": True,
        },
        "phrase": {
            "type": "string",
            "description": "Optional description of the one object to place inside box.",
            "nullable": True,
        },
    },
    "image",
)
def generate_object(
    models: ModelCache,
    image: Image.Image,
    prompt: str,
    mask: Image.Image | None = None,
    box: list[float] | None = None,
    phrase: str | None = None,
) -> Image.Image:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if phrase is not None:
        if not phrase.strip():
            raise ValueError("phrase must be a non-empty string")
        if mask is not None:
            raise ValueError("Phrase-guided generation requires a box, not a mask")
        if box is None:
            raise ValueError("Phrase-guided generation requires a box")
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(validate_box(box), fill=255)
        prompt = f"{prompt.strip()} Place {phrase.strip()} inside the selected region."
        return edit_masked_object(
            models,
            image,
            mask,
            prompt,
            mode="regional",
            padding=0,
        )

    if mask is not None and box is not None:
        raise ValueError("Provide a mask or a box, not both")
    if mask is None and box is None:
        raise ValueError("Provide one mask or one box to select a generation region")
    if mask is None:
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(validate_box(box), fill=255)
    return edit_masked_object(
        models,
        image,
        mask,
        prompt,
        mode="regional",
        padding=0,
    )