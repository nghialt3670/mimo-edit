"""Remove an object with FLUX.2 Klein inpainting and the edit-consistency LoRA."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import maximum_filter

from framework.inference.flux2_klein.lora import (
    CONSISTENCY_ADAPTER_NAME,
    CONSISTENCY_LORA_FILE,
    CONSISTENCY_LORA_REPO,
    generate_klein,
)
from framework.inference.flux2_klein.utils import WORK_SIZE, binary_mask
from framework.inference.pool import ModelPool
from framework.inference.utils import validate_number_range

REMOVE_PROMPT = (
    "Remove only the target object and fill the area naturally. "
    "Preserve everything else."
)


def flux2_klein_consistency_lora_inpaint_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    *,
    mode: Literal["remove", "add", "edit", "replace"] = "remove",
    protection_mask: Image.Image | None = None,
    context_padding: float = 1.5,
    mask_padding_ratio: float = 0.05,
    lora_strength: float = 0.6,
    feather_px: int = 24,
    seed: int | None = None,
) -> Image.Image:
    """Inpaint the white region of ``mask`` with the consistency LoRA.

    Only ``mode="remove"`` is implemented; other modes raise ``NotImplementedError``.
    The Klein inpaint pipeline runs on a crop around the mask, and the result is
    pasted back through a feathered mask so pixels away from the object are unchanged.
    ``protection_mask`` marks pixels that must stay original.
    """
    if mode != "remove":
        raise NotImplementedError(f"mode {mode!r} is not implemented yet")

    # --- Validation ---------------------------------------------------------
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("image and mask must be Pillow images")
    if image.size != mask.size:
        raise ValueError("mask must have the same size as image")
    if protection_mask is not None and (
        not isinstance(protection_mask, Image.Image)
        or protection_mask.size != image.size
    ):
        raise ValueError(
            "protection_mask must be a Pillow image the same size as image"
        )
    validate_number_range("context_padding", context_padding, 0, 4)
    validate_number_range("mask_padding_ratio", mask_padding_ratio, 0, 0.5)
    validate_number_range("lora_strength", lora_strength, 0, 1)
    if not isinstance(feather_px, int) or not 0 <= feather_px <= 64:
        raise ValueError("feather_px must be an integer from 0 to 64")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")

    # --- Build the edit mask (dilated, minus protected pixels) ---------------
    original = image.convert("RGB")
    selected = binary_mask(mask).astype(np.uint8) * 255
    bounds = Image.fromarray(selected, "L").getbbox()
    if bounds is None:
        raise ValueError("the mask is empty")
    left, top, right, bottom = bounds
    obj_size = max(right - left, bottom - top)

    radius = round(obj_size * mask_padding_ratio)
    if radius:
        selected = maximum_filter(selected, size=2 * radius + 1, mode="constant")
    protected = binary_mask(protection_mask) if protection_mask is not None else None
    if protected is not None:
        selected[protected] = 0
    edit_mask = Image.fromarray(selected, "L")
    if edit_mask.getbbox() is None:
        raise ValueError("the mask is fully protected")

    # --- Crop around the object ----------------------------------------------
    pad = max(128, round(obj_size * context_padding))
    box = (
        max(0, left - pad),
        max(0, top - pad),
        min(original.width, right + pad),
        min(original.height, bottom + pad),
    )
    crop = original.crop(box)
    crop_mask = edit_mask.crop(box)

    # --- Run the inpaint pipeline at a VAE-friendly size (multiple of 16) ----
    cw, ch = crop.size
    scale = min(1.0, WORK_SIZE / max(cw, ch))
    ww = max(64, round(cw * scale / 16) * 16)
    wh = max(64, round(ch * scale / 16) * 16)
    work_image = crop.resize((ww, wh), Image.Resampling.LANCZOS)
    work_mask = crop_mask.resize((ww, wh), Image.Resampling.NEAREST)

    result = generate_klein(
        pool.flux2_klein_inpaint_pipeline,
        work_image,
        REMOVE_PROMPT,
        pool.config.klein,
        adapter_name=CONSISTENCY_ADAPTER_NAME,
        adapter_weight=lora_strength,
        pretrained=CONSISTENCY_LORA_REPO,
        weight_name=CONSISTENCY_LORA_FILE,
        seed=seed,
        height=wh,
        width=ww,
        mask_image=work_mask,
    )
    if result.size != (cw, ch):
        result = result.resize((cw, ch), Image.Resampling.LANCZOS)

    # --- Composite back through a feathered mask -----------------------------
    blend = crop_mask
    if feather_px:
        grown = maximum_filter(
            np.asarray(crop_mask), size=2 * (feather_px // 2) + 1, mode="constant"
        )
        soft = Image.fromarray(grown, "L").filter(
            ImageFilter.GaussianBlur(radius=feather_px / 2)
        )
        blend = ImageChops.lighter(crop_mask, soft)
    if protected is not None:
        arr = np.asarray(blend).copy()
        arr[protected[box[1] : box[3], box[0] : box[2]]] = 0
        blend = Image.fromarray(arr, "L")

    output = original.copy()
    output.paste(result, box[:2], blend)
    return output
