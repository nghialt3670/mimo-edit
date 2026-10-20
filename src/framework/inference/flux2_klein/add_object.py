"""Insert a new object into an image with FLUX.2 Klein."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from framework.utils.randomness import get_random_seed

from ..pool import ModelPool
from .utils import (
    PADDING_FRAC,
    WORK_SIZE,
    binary_mask,
    blend_edit,
    generate_klein_edit,
    segment_added_object,
)


def flux2_klein_add_object_with_box(
    pool: ModelPool,
    image: Image.Image,
    box: Sequence[float],
    prompt: str,
    debug: dict | None = None,
) -> Image.Image:
    """Add an object near an XYXY pixel box.

    Klein edits the box crop; SAM3 segments the added object in its output (box
    prompt), and the ViTMatte matte seeded by that mask is the paste weight.
    """
    pipe = pool.flux2_klein_pipeline
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("box must contain four finite XYXY coordinates")
    original = image.convert("RGB")
    width, height = original.size
    x1, y1, x2, y2 = map(
        int,
        np.clip(
            [round(float(v)) for v in box],
            [0, 0, 0, 0],
            [width, height, width, height],
        ),
    )
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box must have positive area inside the image")
    pad = math.ceil(max(x2 - x1, y2 - y1) * PADDING_FRAC)
    # Slide the window back inside the image instead of edge-padding it, so
    # Klein sees real context when the box touches a border.
    cx1, cy1 = x1 - pad, y1 - pad
    cx2, cy2 = x2 + pad, y2 + pad
    shift_x = max(0, -cx1) - max(0, cx2 - width)
    shift_y = max(0, -cy1) - max(0, cy2 - height)
    crop_box = (
        max(0, cx1 + shift_x),
        max(0, cy1 + shift_y),
        min(width, cx2 + shift_x),
        min(height, cy2 + shift_y),
    )
    cropped = original.crop(crop_box)
    cw, ch = cropped.size
    scale = WORK_SIZE / max(cw, ch)
    ww = max(64, round(cw * scale / 16) * 16)
    wh = max(64, round(ch * scale / 16) * 16)
    with torch.inference_mode():
        inference_config = pool.config.klein
        edited = (
            pipe(
                image=cropped.resize((ww, wh), Image.Resampling.LANCZOS),
                prompt=f"Add {prompt} that should be big and fit the whole image, preserving the surrounding context.",
                height=wh,
                width=ww,
                num_inference_steps=inference_config.num_inference_steps,
                guidance_scale=inference_config.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(get_random_seed()),
            )
            .images[0]
            .convert("RGB")
        )
    if edited.size != (ww, wh):
        raise ValueError(f"Klein returned {edited.size}, expected {(ww, wh)}")
    edited = edited.resize((cw, ch), Image.Resampling.LANCZOS)
    box_mask = np.zeros((ch, cw), bool)
    box_mask[
        y1 - crop_box[1] : y2 - crop_box[1], x1 - crop_box[0] : x2 - crop_box[0]
    ] = True
    object_mask = segment_added_object(pool, cropped, edited, box_mask)
    result, matte = blend_edit(
        pool, original, crop_box, edited, object_mask, debug=debug
    )
    if debug is not None:
        debug.update(before=cropped, klein=edited, matte=matte, minimum=object_mask)
    return result


def flux2_klein_add_object_with_mask(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    debug: dict | None = None,
) -> Image.Image:
    """Add an object at a white mask.

    Klein edits a crop around the mask; SAM3 segments the added object in its
    output, and the ViTMatte matte seeded by that mask is the paste weight.
    """
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("image and mask must both be PIL images")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if mask.size != image.size:
        raise ValueError("mask must have the same size as image")
    original = image.convert("RGB")
    mask = Image.fromarray(binary_mask(mask).astype(np.uint8) * 255, "L")
    region = mask.getbbox()
    if region is None:
        raise ValueError("mask is empty")
    inference_config = pool.config.klein
    left, top, right, bottom = region
    padding = inference_config.mask_context_padding
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(original.width, right + padding),
        min(original.height, bottom + padding),
    )
    original_crop = original.crop(crop_box)
    working_image = ImageOps.contain(original_crop, (WORK_SIZE, WORK_SIZE))
    edited = generate_klein_edit(
        pool.flux2_klein_pipeline,
        working_image,
        f"Add {prompt} inside the selected region. Preserve the surroundings.",
        inference_config,
    )
    if edited.size != working_image.size:
        edited = edited.resize(working_image.size, Image.Resampling.LANCZOS)
    edited = edited.resize(original_crop.size, Image.Resampling.LANCZOS)
    object_mask = binary_mask(mask.crop(crop_box))
    object_mask = segment_added_object(pool, original_crop, edited, object_mask)
    result, matte = blend_edit(
        pool, original, crop_box, edited, object_mask, debug=debug
    )
    if debug is not None:
        debug.update(
            before=original_crop, klein=edited, matte=matte, minimum=object_mask
        )
    return result
