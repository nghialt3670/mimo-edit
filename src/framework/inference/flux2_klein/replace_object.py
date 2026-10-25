"""Replace an object with a new one in a single FLUX.2 Klein pass."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..pool import ModelPool
from .placement import GUIDE
from .utils import (
    WORK_SIZE,
    binary_mask,
    blend_edit,
    generate_klein_edit,
    segment_added_object,
)

REPLACE_PROMPT = "Replace the object in the middle of the image with {prompt}. {guide} Add nothing else and preserve the surroundings."
# The old object's area (grown by this fraction of its size) takes Klein's pixels.
OLD_AREA_GROW = 0.04


def _old_area_weight(mask: Image.Image, crop_box) -> np.ndarray:
    """Soft weight on the old object's footprint, so none of it is left behind."""
    left, top, right, bottom = mask.getbbox()
    grow = max(2, round(OLD_AREA_GROW * max(right - left, bottom - top)))
    grown = mask.crop(crop_box).filter(ImageFilter.MaxFilter(2 * grow + 1))
    soft = grown.filter(ImageFilter.GaussianBlur(grow / 2))
    return np.asarray(soft, np.float32) / 255.0


def flux2_klein_replace_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    debug: dict | None = None,
    *,
    refine: bool = False,
) -> Image.Image:
    """Replace the object at a white mask with ``prompt``, without removing it first.

    Klein edits a crop around the mask and replaces the object directly. SAM3 and
    the change map segment the new object, ViTMatte gives its alpha, and the old
    object's footprint also takes Klein's pixels.
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
    old_mask = binary_mask(mask.crop(crop_box))

    out = generate_klein_edit(
        pool.flux2_klein_pipeline,
        working_image,
        REPLACE_PROMPT.format(prompt=prompt, guide=GUIDE),
        inference_config,
    )
    if out.size != working_image.size:
        out = out.resize(working_image.size, Image.Resampling.LANCZOS)
    edited = out.resize(original_crop.size, Image.Resampling.LANCZOS)
    object_mask = segment_added_object(
        pool, original_crop, edited, old_mask, prompt, debug=debug
    )
    result, matte = blend_edit(
        pool,
        original,
        crop_box,
        edited,
        object_mask,
        harmonize=refine,
        extra_weight=_old_area_weight(mask, crop_box),
        debug=debug,
    )
    if debug is not None:
        debug.update(
            before=original_crop,
            klein=edited,
            matte=matte,
            minimum=object_mask,
        )
    return result
