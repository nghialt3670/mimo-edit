from __future__ import annotations

from PIL import Image

from framework.inference.object_clear import object_clear_remove_object

from ..context import ToolContext
from ..registry import register_tool
from ..utils import (
    masked_layer,
    paste_layer,
    validate_box,
    validate_image,
    validate_mask,
    validate_number_list,
)


def _validate_move_object_args(
    image: Image.Image,
    mask: Image.Image,
    target: list[float],
) -> None:
    validate_image("image", image)
    validate_mask(image, mask)
    coordinates = validate_number_list("target", target, lengths=(2, 4))
    if len(coordinates) == 4 and (
        coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]
    ):
        raise ValueError("target box must have positive width and height")


@register_tool(
    "move_object",
    "Move the selected object to a new location.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {"type": "image", "description": "Mask selecting the object."},
        "target": {
            "type": "array",
            "description": "Destination pixel [x, y] or box [x1, y1, x2, y2].",
        },
    },
    "image",
)
def move_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    target: list[float],
) -> Image.Image:
    _validate_move_object_args(image, mask, target)
    layer, _ = masked_layer(image, mask)
    if len(target) == 2:
        destination = (
            round(float(target[0]) - layer.width / 2),
            round(float(target[1]) - layer.height / 2),
        )
    else:
        x1, y1, x2, y2 = validate_box(target)
        factor = min((x2 - x1) / layer.width, (y2 - y1) / layer.height)
        layer = layer.resize(
            (max(1, round(layer.width * factor)), max(1, round(layer.height * factor))),
            Image.Resampling.LANCZOS,
        )
        destination = (
            round(x1 + (x2 - x1 - layer.width) / 2),
            round(y1 + (y2 - y1 - layer.height) / 2),
        )
    clean = object_clear_remove_object(ctx.model_pool, image, mask)
    return paste_layer(clean, layer, destination)
