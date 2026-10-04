from __future__ import annotations

from PIL import Image

from utils.image import masked_layer, paste_layer
from inference import ModelCache, remove_with_object_clear
from ..registry import register_tool


@register_tool(
    "flip_object",
    "Flip one masked object horizontally ('x') or vertically ('y'), preserving its position.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask selecting the object."},
        "axis": {"type": "string", "description": "'x' for horizontal or 'y' for vertical."},
    },
    "image",
)
def flip_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    axis: str,
) -> Image.Image:
    transpose = {
        "x": Image.Transpose.FLIP_LEFT_RIGHT,
        "y": Image.Transpose.FLIP_TOP_BOTTOM,
    }
    if axis not in transpose:
        raise ValueError("axis must be 'x' or 'y'")
    layer, origin = masked_layer(image, mask)
    clean = remove_with_object_clear(
        models, image, mask, "remove the selected object", model_key="flip_object"
    )
    return paste_layer(clean, layer.transpose(transpose[axis]), origin)