"""Remove an object with FLUX.2 Klein and a selectable removal LoRA."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import maximum_filter

from ..pool import ModelPool
from ..utils import validate_number_range
from .lora import (
    REMOVE_LORAS,
    generate_klein,
    load_remove_prompt_embeds,
    remove_lora_spec,
    validated_object_remove_lora_path,
)
from .shadow import find_shadow_mask
from .utils import WORK_SIZE, harmonize_with_mask

REMOVE_PROMPT = REMOVE_LORAS["fal"].prompt
HIGHLIGHT_COLOR = (255, 0, 255)


def _poisson_blend(
    result: Image.Image, reference: Image.Image, mask: Image.Image, grow_px: int
) -> Image.Image:
    """Blend ``result`` into ``reference`` with Poisson (gradient-domain) cloning.

    Inside the mask the generated texture is kept, but its colors are solved so
    they match ``reference`` exactly on the mask boundary. This removes the tone
    jump (VAE drift, PCT-Net color change) that feathering alone cannot hide.
    """
    selected = (np.asarray(mask) > 127).astype(np.uint8)
    if grow_px:
        selected = maximum_filter(selected, size=2 * grow_px + 1, mode="constant")
    if not selected.any():
        return result
    # Cloning needs the mask boundary strictly inside the image, and it keeps the
    # boundary pixels equal to the destination. Pad all three by a margin so an
    # object touching the image edge is still cloned right up to the edge; in the
    # margin the destination is the generated fill itself, so the edge stays
    # consistent with it instead of reverting to the original object.
    pad = 8
    source = np.pad(np.asarray(result), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    destination = np.pad(
        np.asarray(reference), ((pad, pad), (pad, pad), (0, 0)), mode="edge"
    )
    clone = np.pad(selected, pad, mode="edge")
    clone[[0, -1], :] = 0
    clone[:, [0, -1]] = 0
    margin = np.ones(clone.shape, bool)
    margin[pad:-pad, pad:-pad] = False
    destination[margin & (clone > 0)] = source[margin & (clone > 0)]
    ys, xs = np.nonzero(clone)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    try:
        blended = cv2.seamlessClone(
            np.ascontiguousarray(source[y0:y1, x0:x1]),
            np.ascontiguousarray(destination),
            np.ascontiguousarray(clone[y0:y1, x0:x1] * 255),
            (int(x0 + (x1 - x0) // 2), int(y0 + (y1 - y0) // 2)),
            cv2.NORMAL_CLONE,
        )
    except cv2.error:
        return result
    return Image.fromarray(blended[pad:-pad, pad:-pad])


def _binary(mask: Image.Image) -> np.ndarray:
    array = np.asarray(mask.convert("L"))
    threshold = 1 if int(array.max()) <= 1 else 128
    return (array >= threshold).astype(np.uint8) * 255


def _classical_fill(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Erase the object with OpenCV inpainting so no trace of it reaches Klein."""
    filled = cv2.inpaint(np.asarray(image), _binary(mask), 5, cv2.INPAINT_TELEA)
    return Image.fromarray(filled)


def _highlight(image: Image.Image, mask: Image.Image, opacity: float) -> Image.Image:
    """Paint the masked target magenta, the marking the LoRA is prompted with."""
    rgb = np.asarray(image, dtype=np.float32).copy()
    mark = np.asarray(mask) > 127
    highlight = np.array(HIGHLIGHT_COLOR, dtype=np.float32)
    rgb[mark] = (1 - opacity) * rgb[mark] + opacity * highlight
    return Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGB")


def flux2_klein_remove_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    *,
    protection_mask: Image.Image | None = None,
    context_padding: float = 1.5,
    mask_padding_ratio: float = 0.05,
    prompt: str | None = None,
    highlight_opacity: float = 0.65,
    lora: str | None = None,
    lora_scale: float | None = None,
    lora_repo: str | None = None,
    lora_weight: str | None = None,
    feather_px: int = 24,
    harmonize: bool = False,
    include_shadow: bool = False,
    seed: int | None = None,
) -> Image.Image:
    """Remove the white region of ``mask``.

    ``lora`` selects the removal adapter: ``fal`` (magenta highlight on a
    Telea fill) or ``slopshop`` (erase-v1: original photo plus RGB mask).
    When omitted, ``pool.config.klein.remove_lora`` is used. Each adapter is
    loaded once under its own name and switched on for this call.

    1. Crop the image around the mask (plus context). With ``include_shadow``
       the object's cast shadow (SAM3 "shadow", or a luminance heuristic as
       fallback) is added to the removal mask.
    2. Run Klein with the selected LoRA protocol.
    3. Harmonize the filled mask region with PCT-Net (``harmonize``).
    4. Poisson-blend the patch into the original so its boundary has no tone jump.
    5. Paste the result back through the feathered mask, so pixels away from the
       object are never changed. ``protection_mask`` marks pixels that must stay
       original even when they overlap the dilated mask.
    """
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
    spec = remove_lora_spec(lora if lora is not None else pool.config.klein.remove_lora)
    adapter_weight = spec.scale if lora_scale is None else lora_scale
    repo = spec.repo if lora_repo is None else lora_repo
    weight_name = spec.weight if lora_weight is None else lora_weight
    instruction = spec.prompt if prompt is None else prompt
    validate_number_range("context_padding", context_padding, 0, 4)
    validate_number_range("mask_padding_ratio", mask_padding_ratio, 0, 0.5)
    validate_number_range("lora_scale", adapter_weight, 0, 2)
    validate_number_range("highlight_opacity", highlight_opacity, 0.1, 1.0)
    if not isinstance(feather_px, int) or not 0 <= feather_px <= 64:
        raise ValueError("feather_px must be an integer from 0 to 64")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")

    # --- Build the edit mask (dilated, minus protected pixels) ---------------
    original = image.convert("RGB")
    selected = _binary(mask)
    bounds = Image.fromarray(selected, "L").getbbox()
    if bounds is None:
        raise ValueError("the mask is empty")
    left, top, right, bottom = bounds
    obj_size = max(right - left, bottom - top)

    radius = round(obj_size * mask_padding_ratio)
    if radius:
        selected = maximum_filter(selected, size=2 * radius + 1, mode="constant")
    protected = _binary(protection_mask) > 0 if protection_mask is not None else None
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
    extra = round(spec.mask_grow_frac * max(crop.size))
    if extra:
        crop_mask = Image.fromarray(
            maximum_filter(np.asarray(crop_mask), size=2 * extra + 1, mode="constant"),
            "L",
        )
    if include_shadow:
        shadow = find_shadow_mask(pool, crop, _binary(mask.crop(box)))
        if protected is not None:
            shadow[protected[box[1] : box[3], box[0] : box[2]]] = 0
        crop_mask = Image.fromarray(np.maximum(np.asarray(crop_mask), shadow), "L")

    # --- Run the LoRA edit at a VAE-friendly size (multiple of 16) -----------
    cw, ch = crop.size
    scale = min(1.0, WORK_SIZE / max(cw, ch))
    ww = max(64, round(cw * scale / 16) * 16)
    wh = max(64, round(ch * scale / 16) * 16)
    work_image = crop.resize((ww, wh), Image.Resampling.LANCZOS)
    work_mask = crop_mask.resize((ww, wh), Image.Resampling.NEAREST)

    if spec.highlight:
        klein_image: Image.Image | list[Image.Image] = _highlight(
            _classical_fill(work_image, work_mask), work_mask, highlight_opacity
        )
    else:
        klein_image = [work_image, work_mask.convert("RGB")]
    embeds = load_remove_prompt_embeds(spec) if prompt is None else None
    checkpoint = validated_object_remove_lora_path(
        repo, weight_name, validate_comfy_keys=spec.validate_comfy_keys
    )
    result = generate_klein(
        pool.flux2_klein_pipeline,
        klein_image,
        None if embeds is not None else instruction,
        pool.config.klein,
        prompt_embeds=embeds,
        adapter_name=spec.adapter_name,
        adapter_weight=adapter_weight,
        pretrained=str(checkpoint.parent),
        weight_name=checkpoint.name,
        seed=seed,
        height=wh,
        width=ww,
    )
    if result.size != (cw, ch):
        result = result.resize((cw, ch), Image.Resampling.LANCZOS)

    # --- Composite back through a feathered mask -----------------------------
    if harmonize:
        result = harmonize_with_mask(pool.pctnet_pipeline, result, crop_mask)
    # PCT-Net hard-composites inside the mask, so its color change ends abruptly
    # at the mask edge; Poisson blending makes the patch continuous with the
    # original pixels around it.
    result = _poisson_blend(result, crop, crop_mask, grow_px=max(2, feather_px // 4))
    blend = crop_mask
    if feather_px:
        # Grow the mask by the feather width, then blur: the fill stays solid over
        # the object and fades out smoothly well beyond its edge.
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
