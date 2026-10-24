from __future__ import annotations

from PIL import Image

from framework.inference import qwen_locate_region

from ..context import ToolContext
from ..registry import register_tool
from ..utils import validate_image, validate_text


@register_tool(
    "locate_region",
    "Locate a region described in natural language and return its pixel bounding box.",
    {
        "image": {
            "type": "image",
            "description": "Image containing the region.",
        },
        "prompt": {
            "type": "string",
            "description": (
                "Natural-language description of the region to locate."
            ),
        },
    },
    "array",
)
def locate_region(
    ctx: ToolContext,
    image: Image.Image,
    prompt: str,
) -> list[float]:
    validate_image("image", image)
    validate_text("prompt", prompt)
    return qwen_locate_region(ctx.model_pool, image, prompt)
