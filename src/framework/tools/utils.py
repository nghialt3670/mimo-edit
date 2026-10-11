from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import distance_transform_edt

_EDGE_FEATHER_RADIUS = 1.0
_EDGE_FEATHER_SUPPORT = 3


def validate_image(name: str, value: object) -> Image.Image:
    if not isinstance(value, Image.Image):
        raise TypeError(f"{name} must be a Pillow image")
    return value


def validate_mask(
    image: Image.Image,
    mask: object,
    *,
    name: str = "mask",
    same_size: bool = False,
) -> Image.Image:
    mask = validate_image(name, mask)
    if same_size and mask.size != image.size:
        raise ValueError(f"{name} must have the same dimensions as image")
    normalized = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    if normalized.getbbox() is None:
        raise ValueError(f"{name} must contain a non-empty region")
    return mask


def validate_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def validate_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def validate_positive_number(name: str, value: object) -> None:
    if validate_number(name, value) <= 0:
        raise ValueError(f"{name} must be greater than zero")


def validate_number_range(
    name: str,
    value: object,
    minimum: float,
    maximum: float,
    *,
    maximum_inclusive: bool = True,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (value > maximum if maximum_inclusive else value >= maximum)
    ):
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")


def validate_box(values: list[float]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    result = tuple(round(float(value)) for value in values)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box must have positive width and height")
    return result


def validate_number_list(
    name: str,
    value: object,
    *,
    lengths: tuple[int, ...] = (2,),
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numeric list")
    if len(value) not in lengths:
        expected = " or ".join(map(str, lengths))
        raise ValueError(f"{name} must contain {expected} values")
    return tuple(validate_number(f"{name} values", item) for item in value)


def masked_layer(
    image: Image.Image, mask: Image.Image
) -> tuple[Image.Image, tuple[int, int]]:
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("mask is empty")
    left, top, right, bottom = bounds
    crop_box = (
        max(0, left - _EDGE_FEATHER_SUPPORT),
        max(0, top - _EDGE_FEATHER_SUPPORT),
        min(image.width, right + _EDGE_FEATHER_SUPPORT),
        min(image.height, bottom + _EDGE_FEATHER_SUPPORT),
    )
    mask_crop = mask.crop(crop_box)
    mask_array = np.asarray(mask_crop)
    foreground = mask_array > 0
    nearest_indices = distance_transform_edt(
        ~foreground, return_distances=False, return_indices=True
    )
    image_crop = image.convert("RGB").crop(crop_box)
    image_array = np.asarray(image_crop).copy()
    nearest_colors = image_array[tuple(nearest_indices)]
    image_array[~foreground] = nearest_colors[~foreground]
    feathered_alpha = mask_crop.filter(
        ImageFilter.GaussianBlur(radius=_EDGE_FEATHER_RADIUS)
    )
    layer = Image.fromarray(image_array, mode="RGB").convert("RGBA")
    layer.putalpha(feathered_alpha)
    return layer, crop_box[:2]


def paste_layer(
    image: Image.Image, layer: Image.Image, position: tuple[int, int]
) -> Image.Image:
    result = image.convert("RGBA")
    result.alpha_composite(layer.convert("RGBA"), dest=position)
    return result.convert("RGB")


def composite_masked(
    image: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
    *,
    feather_radius: float = 1.0,
) -> Image.Image:
    if image.size != edited.size:
        raise ValueError("image and edited image must have the same size")
    if feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")
    selection = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    if selection.getbbox() is None:
        raise ValueError("mask is empty")
    if feather_radius:
        softened = selection.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        selection = ImageChops.darker(selection, softened)
    return Image.composite(edited.convert("RGB"), image.convert("RGB"), selection)
