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
    edge_padded_crop,
    generate_klein_edit,
)


def flux2_klein_add_object_with_box(
    pool: ModelPool,
    image: Image.Image,
    box: Sequence[float],
    prompt: str,
    *,
    harmonize: bool = True,
) -> Image.Image:
    """Add an object near an XYXY pixel box.

    Klein edits the box crop; ``blend_edit`` then pastes back only the pixels it
    changed (the new object), harmonized with PCT-Net (``harmonize``) with a per-pixel alpha.
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
    crop_box = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    cropped = edge_padded_crop(original, crop_box)
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
    blended = blend_edit(
        cropped,
        edited,
        minimum=np.zeros((ch, cw), bool),
        limit=np.ones((ch, cw), bool),
        predictor=pool.pctnet_pipeline if harmonize else None,
    )
    cx1, cy1, cx2, cy2 = crop_box
    vx1, vy1 = max(cx1, 0), max(cy1, 0)
    vx2, vy2 = min(cx2, width), min(cy2, height)
    result = original.copy()
    result.paste(
        blended.crop((vx1 - cx1, vy1 - cy1, vx2 - cx1, vy2 - cy1)),
        (vx1, vy1),
    )
    return result


def flux2_klein_add_object_with_mask(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    harmonize: bool = True,
) -> Image.Image:
    """Add an object at a white mask.

    Klein edits a crop around the mask; ``blend_edit`` then pastes back the mask
    plus any significant change connected to it, harmonized with PCT-Net
    (``harmonize``) with a per-pixel alpha.
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
    blended = blend_edit(
        original_crop,
        edited,
        minimum=object_mask,
        limit=np.ones_like(object_mask),
        predictor=pool.pctnet_pipeline if harmonize else None,
    )
    result = original.copy()
    result.paste(blended, crop_box[:2])
    return result
