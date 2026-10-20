"""Insert a new object into an image with FLUX.2 Klein."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from framework.utils.randomness import get_random_seed

from ..pool import ModelPool
from .placement import GUIDE, placement
from .utils import (
    PADDING_FRAC,
    WORK_SIZE,
    NothingAdded,
    binary_mask,
    blend_edit,
    generate_klein_edit,
    segment_added_object,
)

MIN_ASPECT = 0.5
ATTEMPT_PROMPTS = (
    "Add {prompt}{where} that should be big and fit the whole image. {guide} Preserve the surrounding context.",
    "Clearly draw {prompt}{where} as a distinct, fully visible object. {guide} Preserve the surrounding context.",
)
MASK_ATTEMPT_PROMPTS = (
    "Add {prompt}{where} inside the selected region. {guide} Preserve the surroundings.",
    "Clearly draw {prompt}{where} as a distinct, fully visible object inside the selected region. {guide} Preserve the surroundings.",
)


def _paste(
    pool, original, crop_box, edited, object_mask, refine, debug, extra_weight=None
):
    """Paste with the matte, then harmonize with PCT-Net or refine with Klein."""
    result, matte = blend_edit(
        pool,
        original,
        crop_box,
        edited,
        object_mask,
        harmonize=not refine,
        extra_weight=extra_weight,
        debug=debug,
    )
    if refine:
        from .refine import klein_refine

        full = np.zeros((original.height, original.width), np.float32)
        left, top, right, bottom = crop_box
        full[top:bottom, left:right] = matte
        result, raw = klein_refine(pool, result, full)
        if debug is not None:
            debug["refined"] = result
            debug["refined_raw"] = raw
    return result, matte


def _generate_with_retry(pool, before, region, prompt, generate, debug):
    """Generate and segment the new object, retrying once with another seed and wording."""
    error: NothingAdded | None = None
    for attempt in range(len(ATTEMPT_PROMPTS)):
        edited = generate(attempt)
        try:
            mask = segment_added_object(
                pool, before, edited, region, prompt, debug=debug
            )
        except NothingAdded as exc:
            error = exc
            continue
        if debug is not None:
            debug["attempt"] = attempt
        return edited, mask
    raise NothingAdded(f"{error} (tried {len(ATTEMPT_PROMPTS)} times: {prompt!r})")


def flux2_klein_add_object_with_box(
    pool: ModelPool,
    image: Image.Image,
    box: Sequence[float],
    prompt: str,
    debug: dict | None = None,
    *,
    anchor: str | None = None,
    refine: bool = False,
) -> Image.Image:
    """Add an object near an XYXY pixel box.

    ``anchor`` is a placement phrase ("on the blue cooler") added to the prompt.
    With ``refine`` the pasted result is run through Klein once more on the whole
    image instead of PCT-Net, to blend the object in (see ``klein_refine``).

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
    # Klein handles very flat or very thin crops badly, so grow the short side.
    grow_x = max(0, math.ceil((cy2 - cy1) * MIN_ASPECT) - (cx2 - cx1))
    grow_y = max(0, math.ceil((cx2 - cx1) * MIN_ASPECT) - (cy2 - cy1))
    cx1, cx2 = cx1 - grow_x // 2, cx2 + grow_x - grow_x // 2
    cy1, cy2 = cy1 - grow_y // 2, cy2 + grow_y - grow_y // 2
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
    box_mask = np.zeros((ch, cw), bool)
    box_mask[
        y1 - crop_box[1] : y2 - crop_box[1], x1 - crop_box[0] : x2 - crop_box[0]
    ] = True
    inference_config = pool.config.klein
    working = cropped.resize((ww, wh), Image.Resampling.LANCZOS)

    def generate(attempt: int) -> Image.Image:
        with torch.inference_mode():
            out = (
                pipe(
                    image=working,
                    prompt=ATTEMPT_PROMPTS[attempt].format(
                        prompt=prompt, where=placement(anchor), guide=GUIDE
                    ),
                    height=wh,
                    width=ww,
                    num_inference_steps=inference_config.num_inference_steps,
                    guidance_scale=inference_config.guidance_scale,
                    generator=torch.Generator(device="cpu").manual_seed(
                        get_random_seed() + attempt
                    ),
                )
                .images[0]
                .convert("RGB")
            )
        if out.size != (ww, wh):
            raise ValueError(f"Klein returned {out.size}, expected {(ww, wh)}")
        return out.resize((cw, ch), Image.Resampling.LANCZOS)

    edited, object_mask = _generate_with_retry(
        pool, cropped, box_mask, prompt, generate, debug
    )
    result, matte = _paste(pool, original, crop_box, edited, object_mask, refine, debug)
    if debug is not None:
        debug.update(
            before=cropped,
            klein=edited,
            matte=matte,
            minimum=object_mask,
            anchor=anchor,
        )
    return result


def flux2_klein_add_object_with_mask(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    debug: dict | None = None,
    *,
    anchor: str | None = None,
    refine: bool = False,
) -> Image.Image:
    """Add an object at a white mask.

    ``anchor`` and ``refine`` work as in ``flux2_klein_add_object_with_box``.

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
    object_mask = binary_mask(mask.crop(crop_box))

    def generate(attempt: int) -> Image.Image:
        out = generate_klein_edit(
            pool.flux2_klein_pipeline,
            working_image,
            MASK_ATTEMPT_PROMPTS[attempt].format(
                prompt=prompt, where=placement(anchor), guide=GUIDE
            ),
            inference_config,
            seed_offset=attempt,
        )
        if out.size != working_image.size:
            out = out.resize(working_image.size, Image.Resampling.LANCZOS)
        return out.resize(original_crop.size, Image.Resampling.LANCZOS)

    edited, object_mask = _generate_with_retry(
        pool, original_crop, object_mask, prompt, generate, debug
    )
    result, matte = _paste(pool, original, crop_box, edited, object_mask, refine, debug)
    if debug is not None:
        debug.update(
            before=original_crop,
            klein=edited,
            matte=matte,
            minimum=object_mask,
            anchor=anchor,
        )
    return result
