"""Remove an object: classical inpaint fill, then FLUX.2 Klein + removal LoRA."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import maximum_filter

from framework.utils.randomness import get_random_seed

from ..pool import ModelPool
from ..utils import validate_number_range
from .shadow import find_shadow_mask
from .utils import WORK_SIZE, harmonize_with_mask

REMOVE_PROMPT = "Remove the highlighted object from the scene"
HIGHLIGHT_COLOR = (255, 0, 255)

OBJECT_REMOVE_LORA_REPO = "fal/flux-2-klein-4B-object-remove-lora"
# The 58 MB flux-object-remove-lora.safetensors upload is incomplete upstream.
# Use the validated, 76 MB ComfyUI-converted checkpoint instead. Diffusers'
# Flux2 LoRA loader converts its diffusion_model.* keys automatically.
OBJECT_REMOVE_LORA_WEIGHT = (
    "kDEkt5q7tDLKOpQJIVMPx_" "pytorch_lora_weights_comfy_converted.safetensors"
)


def _validated_object_remove_lora_path(repo_or_path: str, weight_name: str) -> Path:
    """Resolve a Hub or local checkpoint and reject unreadable safetensors.

    Do not retry a malformed checkpoint: the original 58 MB upstream upload
    is known to have missing tensor data and redownloading cannot repair it.
    """
    from safetensors import safe_open

    location = Path(repo_or_path).expanduser()
    if location.is_file():
        checkpoint = location
    elif location.is_dir():
        checkpoint = location / weight_name
    else:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(hf_hub_download(repo_id=repo_or_path, filename=weight_name))

    if not checkpoint.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint}")

    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as reader:
            keys = list(reader.keys())
            if not keys:
                raise ValueError("checkpoint contains no tensors")
            if checkpoint.name == OBJECT_REMOVE_LORA_WEIGHT and not all(
                key.startswith("diffusion_model.") for key in keys
            ):
                raise ValueError("unexpected keys in the 76 MB object-removal LoRA")
    except Exception as exc:
        raise RuntimeError(
            f"Unreadable object-removal LoRA checkpoint: {checkpoint}. "
            "The original 58 MB checkpoint is incomplete upstream. "
            f"Use {OBJECT_REMOVE_LORA_WEIGHT} from {OBJECT_REMOVE_LORA_REPO}."
        ) from exc

    return checkpoint


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
    selected[[0, -1], :] = 0  # cloning needs a boundary inside the image
    selected[:, [0, -1]] = 0
    ys, xs = np.nonzero(selected)
    if len(ys) == 0:
        return result
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    source = np.ascontiguousarray(np.asarray(result)[y0:y1, x0:x1])
    clone_mask = np.ascontiguousarray(selected[y0:y1, x0:x1] * 255)
    destination = np.ascontiguousarray(np.asarray(reference))
    try:
        blended = cv2.seamlessClone(
            source,
            destination,
            clone_mask,
            (int(x0 + (x1 - x0) // 2), int(y0 + (y1 - y0) // 2)),
            cv2.NORMAL_CLONE,
        )
    except cv2.error:
        return result
    return Image.fromarray(blended)


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
    prompt: str = REMOVE_PROMPT,
    highlight_opacity: float = 0.65,
    lora_scale: float = 1.3,
    lora_repo: str = OBJECT_REMOVE_LORA_REPO,
    lora_weight: str = OBJECT_REMOVE_LORA_WEIGHT,
    feather_px: int = 24,
    harmonize: bool = False,
    include_shadow: bool = False,
    seed: int | None = None,
) -> Image.Image:
    """Remove the white region of ``mask``.

    1. Crop the image around the mask (plus context). With ``include_shadow``
       the object's cast shadow (SAM3 "shadow", or a luminance heuristic as
       fallback) is added to the removal mask.
    2. Stage 1: erase the mask with OpenCV inpainting, which guarantees no trace
       of the object (even flat ones Klein keeps) but leaves a smeared patch.
    3. Stage 2: highlight the filled area in magenta and run Klein
       with the object-remove LoRA to turn the smear into clean background.
       The LoRA adapter is temporary and deleted in a finally block.
    4. Harmonize the filled mask region with PCT-Net (``harmonize``).
    5. Poisson-blend the patch into the original so its boundary has no tone jump.
    6. Paste the result back through the feathered mask, so pixels away from the
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
    validate_number_range("context_padding", context_padding, 0, 4)
    validate_number_range("mask_padding_ratio", mask_padding_ratio, 0, 0.5)
    validate_number_range("lora_scale", lora_scale, 0, 2)
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

    filled = _classical_fill(work_image, work_mask)
    annotated = _highlight(filled, work_mask, highlight_opacity)

    pipe = pool.flux2_klein_pipeline
    required = (
        "load_lora_weights",
        "set_adapters",
        "delete_adapters",
        "get_active_adapters",
    )
    if any(not callable(getattr(pipe, attr, None)) for attr in required):
        raise RuntimeError(
            "The Klein pipeline needs load_lora_weights, set_adapters, "
            "delete_adapters and get_active_adapters (recent diffusers + peft)."
        )
    if pipe.get_active_adapters():
        raise RuntimeError("The shared Klein pipeline already has active LoRA adapters")
    adapter_name = "_object_remove_temporary"
    checkpoint = _validated_object_remove_lora_path(lora_repo, lora_weight)
    config = pool.config.klein
    try:
        try:
            pipe.load_lora_weights(
                str(checkpoint.parent),
                weight_name=checkpoint.name,
                adapter_name=adapter_name,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Diffusers failed to load the removal LoRA: {checkpoint}"
            ) from exc
        pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
        with torch.inference_mode():
            result = (
                pipe(
                    prompt=prompt,
                    image=annotated,
                    height=wh,
                    width=ww,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=torch.Generator(device="cpu").manual_seed(
                        get_random_seed() if seed is None else seed
                    ),
                )
                .images[0]
                .convert("RGB")
            )
    finally:
        list_adapters = getattr(pipe, "get_list_adapters", None)
        installed = list_adapters() if callable(list_adapters) else {}
        if any(adapter_name in names for names in installed.values()):
            pipe.delete_adapters(adapter_name)
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
