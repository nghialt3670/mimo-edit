from __future__ import annotations

from PIL import Image

from utils.image import masked_layer, paste_layer, validate_box
from inference import ModelCache, remove_with_object_clear
from ..registry import register_tool


def _destination(layer: Image.Image, target: list[float]) -> tuple[Image.Image, tuple[int, int]]:
    if len(target) == 2:
        return layer, (
            round(float(target[0]) - layer.width / 2),
            round(float(target[1]) - layer.height / 2),
        )
    if len(target) == 4:
        x1, y1, x2, y2 = validate_box(target)
        factor = min((x2 - x1) / layer.width, (y2 - y1) / layer.height)
        layer = layer.resize(
            (max(1, round(layer.width * factor)), max(1, round(layer.height * factor))),
            Image.Resampling.LANCZOS,
        )
        return layer, (
            round(x1 + (x2 - x1 - layer.width) / 2),
            round(y1 + (y2 - y1 - layer.height) / 2),
        )
    raise ValueError("each target must be [x, y] or [x1, y1, x2, y2]")


@register_tool(
    "move_object",
    "Move one masked object to a pixel point or box.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask selecting the object."},
        "target": {"type": "array", "description": "Pixel center [x, y] or destination box [x1, y1, x2, y2]."},
    },
    "image",
)
def move_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    target: list[float],
) -> Image.Image:
    layer, _ = masked_layer(image, mask)
    layer, destination = _destination(layer, target)
    clean = remove_with_object_clear(
        models, image, mask, "remove the selected object", model_key="move_object"
    )
    return paste_layer(clean, layer, destination)