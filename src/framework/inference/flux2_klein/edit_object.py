"""Edit an existing object with FLUX.2 Klein."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import maximum_filter

from ..pool import ModelPool
from ..utils import validate_number_range
from .utils import (
    feather_mask,
    generate_klein_edit,
    harmonize_with_mask,
)


def _editing_prompt(
    prompt: str,
    mode: Literal["local", "regional", "global"],
    bounds: tuple[int, int, int, int],
) -> str:
    if mode == "local":
        instructions = (
            "Edit only the selected object's appearance. Preserve its identity, "
            "silhouette, geometry, size, position, and pose. Do not add or remove "
            "parts. Keep everything outside the selected object unchanged."
        )
    elif mode == "regional":
        instructions = (
            "Edit the selected object and allow the requested change to extend into "
            "the padded area around it, including changes to its silhouette, size, "
            "or additions such as accessories. Keep image content outside that "
            "nearby edit area unchanged."
        )
    else:
        instructions = (
            f"The selected object is approximately inside pixel box {list(bounds)}. "
            "Apply the requested edit using the full image for context. Change the "
            "selected object and any surrounding scene elements needed for the "
            "request; preserve unrelated content."
        )
    return f"{instructions}\nRequested edit: {prompt.strip()}"


def flux2_klein_edit_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    mode: Literal["local", "regional", "global"] = "local",
    padding: float = 0.2,
    harmonize: bool = False,
    whole_box: bool = False,
) -> Image.Image:
    """Edit one masked object and feather the local/regional composite boundary.

    With ``whole_box`` (local/regional only), Klein edits the whole bounding box
    of the mask (padded by ``padding`` in regional mode) and the entire edited
    box is pasted back with softened edges instead of only the mask area.

    With ``harmonize`` PCT-Net harmonizes only the object mask region of the
    edited result. It runs before the feathered composite, so the box or
    feathered halo around the mask is never harmonized.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow image")
    if not isinstance(mask, Image.Image):
        raise TypeError("each mask must be a Pillow image")
    if mask.size != image.size:
        raise ValueError("each mask must have the same size as the image")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("each prompt must be a non-empty string")
    if mode not in ("local", "regional", "global"):
        raise ValueError("mode must be 'local', 'regional', or 'global'")
    validate_number_range("padding", padding, 0, 1)
    mask = mask.convert("L")
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("mask is empty")
    left, top, right, bottom = bounds
    current = image.convert("RGB")
    instruction = _editing_prompt(prompt, mode, bounds)
    pipeline = pool.flux2_klein_pipeline
    config = pool.config.klein
    if mode == "global":
        edited = generate_klein_edit(pipeline, current, instruction, config)
        if edited.size != current.size:
            edited = edited.resize(current.size, Image.Resampling.LANCZOS)
        edited = edited.convert("RGB")
        if harmonize:
            edited = harmonize_with_mask(pool.pctnet_pipeline, edited, mask)
        return edited
    if mode == "regional":
        padding_x = round((right - left) * padding)
        padding_y = round((bottom - top) * padding)
        allowed_mask = Image.fromarray(
            maximum_filter(
                np.asarray(mask),
                size=(padding_y * 2 + 1, padding_x * 2 + 1),
                mode="constant",
                cval=0,
            )
        )
    else:
        padding_x = max(16, round((right - left) * 0.2))
        padding_y = max(16, round((bottom - top) * 0.2))
        allowed_mask = mask
    if whole_box and mode != "global":
        box_bounds = allowed_mask.getbbox()
        if box_bounds is None:
            raise ValueError("mask is empty")
        box_left, box_top, box_right, box_bottom = box_bounds
        if mode == "regional":
            box_bounds = (
                max(0, box_left),
                max(0, box_top),
                min(image.width, box_right),
                min(image.height, box_bottom),
            )
        original_crop = current.crop(box_bounds)
        edited_crop = generate_klein_edit(pipeline, original_crop, instruction, config)
        if edited_crop.size != original_crop.size:
            edited_crop = edited_crop.resize(
                original_crop.size, Image.Resampling.LANCZOS
            )
        if harmonize:
            edited_crop = harmonize_with_mask(
                pool.pctnet_pipeline, edited_crop, mask.crop(box_bounds)
            )
        current.paste(
            Image.composite(
                edited_crop, original_crop, feather_mask(original_crop.size)
            ),
            box_bounds[:2],
        )
        return current
    feather_radius = min(12.0, max(2.0, min(right - left, bottom - top) * 0.04))
    feather_support = int(np.ceil(feather_radius * 3))
    allowed_bounds = allowed_mask.getbbox()
    if allowed_bounds is None:
        raise ValueError("mask is empty")
    allowed_left, allowed_top, allowed_right, allowed_bottom = allowed_bounds
    crop_padding_x = feather_support + (padding_x if mode == "local" else 0)
    crop_padding_y = feather_support + (padding_y if mode == "local" else 0)
    crop_box = (
        max(0, allowed_left - crop_padding_x),
        max(0, allowed_top - crop_padding_y),
        min(image.width, allowed_right + crop_padding_x),
        min(image.height, allowed_bottom + crop_padding_y),
    )
    original_crop = current.crop(crop_box)
    feathered_mask = allowed_mask.filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )
    composite_mask = ImageChops.lighter(allowed_mask, feathered_mask)
    edited_crop = generate_klein_edit(pipeline, original_crop, instruction, config)
    if edited_crop.size != original_crop.size:
        edited_crop = edited_crop.resize(original_crop.size, Image.Resampling.LANCZOS)
    if harmonize:
        edited_crop = harmonize_with_mask(
            pool.pctnet_pipeline, edited_crop, mask.crop(crop_box)
        )
    current.paste(edited_crop, crop_box[:2], composite_mask.crop(crop_box))
    return current
