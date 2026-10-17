"""Production editing, insertion, and removal helpers for FLUX.2 Klein 4B."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from scipy.ndimage import maximum_filter

from framework.utils.randomness import get_random_seed

from .config import KleinInferenceConfig
from .pool import ModelPool
from .utils import validate_number_range

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline


def _harmonize_with_mask(predictor, image: Image.Image, mask: Image.Image):
    from .external.pctnet_harmonization import harmonize_pctnet

    return harmonize_pctnet(predictor, image, mask)


def _harmonize_changed_region(
    predictor,
    before: Image.Image,
    after: Image.Image,
    box: Sequence[float],
) -> Image.Image:
    """Harmonize pixels that changed between before/after inside an XYXY box."""
    from .external.pctnet_harmonization import harmonize_pctnet_with_box

    try:
        return harmonize_pctnet_with_box(predictor, before, after, box)
    except ValueError:
        return after


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
    harmonize: bool = True,
    whole_box: bool = False,
) -> Image.Image:
    """Edit one masked object and feather the local/regional composite boundary.

    With ``whole_box`` (local/regional only), Klein edits the whole bounding box
    of the mask (padded by ``padding`` in regional mode) and the entire edited
    box is pasted back with softened edges instead of only the mask area.

    With ``harmonize`` the edit is color-harmonized by PCT-Net afterwards: local
    edits use the object mask; regional and global edits use the pixels changed
    inside the (padded) mask bounds.
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
        edited = _generate_klein_edit(pipeline, current, instruction, config)
        if edited.size != current.size:
            edited = edited.resize(current.size, Image.Resampling.LANCZOS)
        edited = edited.convert("RGB")
        if harmonize:
            edited = _harmonize_changed_region(
                pool.pctnet_pipeline, image.convert("RGB"), edited, bounds
            )
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
        edited_crop = _generate_klein_edit(pipeline, original_crop, instruction, config)
        if edited_crop.size != original_crop.size:
            edited_crop = edited_crop.resize(
                original_crop.size, Image.Resampling.LANCZOS
            )
        original = current.copy()
        current.paste(
            Image.composite(
                edited_crop, original_crop, _feather_mask(original_crop.size)
            ),
            box_bounds[:2],
        )
        if not harmonize:
            return current
        return _harmonize_changed_region(
            pool.pctnet_pipeline, original, current, box_bounds
        )
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
    edited_crop = _generate_klein_edit(pipeline, original_crop, instruction, config)
    if edited_crop.size != original_crop.size:
        edited_crop = edited_crop.resize(original_crop.size, Image.Resampling.LANCZOS)
    original = current.copy()
    current.paste(edited_crop, crop_box[:2], composite_mask.crop(crop_box))
    if not harmonize:
        return current
    if mode == "local":
        return _harmonize_with_mask(pool.pctnet_pipeline, current, mask)
    return _harmonize_changed_region(
        pool.pctnet_pipeline, original, current, allowed_bounds
    )


def _generate_klein_edit(
    pipeline: Flux2KleinPipeline,
    image: Image.Image,
    prompt: str,
    config: KleinInferenceConfig,
) -> Image.Image:
    with torch.inference_mode():
        return (
            pipeline(
                image=image,
                prompt=prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(get_random_seed()),
            )
            .images[0]
            .convert("RGB")
        )


# Shared settings for distilled FLUX.2 Klein 4B.
PADDING_FRAC = 0
WORK_SIZE = 1024
EDGE_BLEND_PX = 12


def _edge_padded_crop(
    image: Image.Image, box: tuple[int, int, int, int]
) -> Image.Image:
    """Crop a possibly out-of-bounds XYXY box, repeating boundary pixels."""
    x1, y1, x2, y2 = box
    w, h = image.size
    valid = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    vx1, vy1, vx2, vy2 = valid
    return Image.fromarray(
        np.pad(
            np.asarray(image.crop(valid)),
            ((vy1 - y1, y2 - vy2), (vx1 - x1, x2 - vx2), (0, 0)),
            mode="edge",
        )
    )


def _feather_mask(size: tuple[int, int]) -> Image.Image:
    """Smoothstep fade at the edges of an edited rectangular crop."""
    width, height = size
    if EDGE_BLEND_PX <= 0:
        return Image.new("L", size, 255)
    x = np.minimum(np.arange(width), np.arange(width)[::-1])[None, :]
    y = np.minimum(np.arange(height), np.arange(height)[::-1])[:, None]
    alpha = np.clip(np.minimum(x, y) / EDGE_BLEND_PX, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    return Image.fromarray(np.uint8(np.rint(alpha * 255)), "L")


def flux2_klein_add_object_with_box(
    pool: ModelPool,
    image: Image.Image,
    box: Sequence[float],
    prompt: str,
    *,
    harmonize: bool = True,
) -> Image.Image:
    """Add an object near an XYXY pixel box and paste the entire edited crop.

    With ``harmonize``, PCT-Net harmonizes the pixels that changed inside the box.
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
    cropped = _edge_padded_crop(original, crop_box)
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
    blended = Image.composite(
        edited.resize((cw, ch), Image.Resampling.LANCZOS),
        cropped,
        _feather_mask((cw, ch)),
    )
    cx1, cy1, cx2, cy2 = crop_box
    vx1, vy1 = max(cx1, 0), max(cy1, 0)
    vx2, vy2 = min(cx2, width), min(cy2, height)
    result = original.copy()
    result.paste(
        blended.crop((vx1 - cx1, vy1 - cy1, vx2 - cx1, vy2 - cy1)),
        (vx1, vy1),
    )
    if harmonize:
        return _harmonize_changed_region(
            pool.pctnet_pipeline, original, result, (x1, y1, x2, y2)
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
    """Generate inside a white mask, preserving every pixel outside the mask.

    With ``harmonize``, PCT-Net harmonizes the masked region.
    """
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("image and mask must both be PIL images")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if mask.size != image.size:
        raise ValueError("mask must have the same size as image")
    original = image.convert("RGB")
    mask = mask.convert("L")
    if mask.getextrema()[1] <= 1:
        mask = mask.point(lambda value: value * 255)
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
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
    edited = _generate_klein_edit(
        pool.flux2_klein_pipeline,
        working_image,
        f"Add {prompt} inside the selected region. Preserve the surroundings.",
        inference_config,
    )
    if edited.size != working_image.size:
        edited = edited.resize(working_image.size, Image.Resampling.LANCZOS)
    crop_mask = mask.crop(crop_box)
    edited_crop = edited.resize(original_crop.size, Image.Resampling.LANCZOS)
    result = original.copy()
    result.paste(edited_crop, crop_box[:2], crop_mask)
    if harmonize:
        return _harmonize_with_mask(pool.pctnet_pipeline, result, mask)
    return result


# LoRA trained to remove objects visibly highlighted in an image (NOT a
# mask-aware inpainting checkpoint). We construct a marked image from a mask.
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


def flux2_klein_remove_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    *,
    protection_mask: Image.Image | None = None,
    context_padding: float = 1.5,
    box_line_px: int | None = None,
    lora_scale: float = 1.0,
    feather_px: int = 10,
    seed: int | None = None,
    lora_repo: str = OBJECT_REMOVE_LORA_REPO,
    lora_weight: str = OBJECT_REMOVE_LORA_WEIGHT,
) -> Image.Image:
    """Remove a highlighted, masked target using the FLUX.2 Klein 4B LoRA.

    White in ``mask`` marks the target. ``protection_mask`` (if supplied)
    marks pixels that MUST remain original, even when near/inside the target.

    The removal LoRA is *image-conditioned*, not a true masked-inpainting
    model. It was trained on images where the target is enclosed by a thin
    pure-red rectangle, so the same annotation is drawn around the mask's
    bounding box. The GENERATED image (never the annotated input) is then
    composited onto the clean original within the target + feathered halo.
    Unprotected pixels within that halo may change. Set ``feather_px=0`` for
    strict outside-mask preservation (at the risk of a visible seam).

    This mutates the shared pooled pipeline ONLY while inference runs:
    the temporary LoRA adapter is deleted in a finally block. Do not call
    other pipeline operations concurrently on the same ModelPool.
    """
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
    if box_line_px is not None and (
        not isinstance(box_line_px, int) or not 1 <= box_line_px <= 32
    ):
        raise ValueError("box_line_px must be an integer from 1 to 32 or None")
    validate_number_range("lora_scale", lora_scale, 0, 2)
    if not isinstance(feather_px, int) or not 0 <= feather_px <= 64:
        raise ValueError("feather_px must be an integer from 0 to 64")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if not lora_repo or not lora_weight:
        raise ValueError("lora_repo and lora_weight must not be empty")

    original = image.convert("RGB")
    selected = np.asarray(mask.convert("L"))
    # Accept both 0..255 masks and 0/1 binary masks.
    threshold = 1 if int(selected.max()) <= 1 else 128
    selected = (selected >= threshold).astype(np.uint8) * 255
    if protection_mask is not None:
        protected = np.asarray(protection_mask.convert("L"))
        protected_threshold = 1 if int(protected.max()) <= 1 else 128
        selected[protected >= protected_threshold] = 0
    binary_mask = Image.fromarray(selected, "L")
    bounds = binary_mask.getbbox()
    if bounds is None:
        raise ValueError("the mask is empty (or fully protected)")

    left, top, right, bottom = bounds
    pad = max(128, round(max(right - left, bottom - top) * context_padding))
    crop_box = (
        max(0, left - pad),
        max(0, top - pad),
        min(original.width, right + pad),
        min(original.height, bottom + pad),
    )
    original_crop = original.crop(crop_box)
    cropped_mask = binary_mask.crop(crop_box)

    # Align dimensions to 16 for stable VAE/transformer processing.
    cw, ch = original_crop.size
    scale = min(1.0, WORK_SIZE / max(cw, ch))
    ww = max(64, round(cw * scale / 16) * 16)
    wh = max(64, round(ch * scale / 16) * 16)
    working_image = original_crop.resize((ww, wh), Image.Resampling.LANCZOS)
    working_mask = cropped_mask.resize((ww, wh), Image.Resampling.NEAREST)

    # The LoRA was trained on images where the target is enclosed by a thin
    # pure-red rectangle (no tint or fill). Anything else, such as a magenta
    # overlay, is out of distribution and the model may paint it into the result.
    annotated = working_image.copy()
    bbox = working_mask.getbbox()
    line = box_line_px or max(3, round(max(ww, wh) / 256))
    margin = line
    ImageDraw.Draw(annotated).rectangle(
        (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(ww - 1, bbox[2] + margin - 1),
            min(wh - 1, bbox[3] + margin - 1),
        ),
        outline=(255, 0, 0),
        width=line,
    )

    pipe = pool.flux2_klein_pipeline
    required = (
        "load_lora_weights",
        "set_adapters",
        "delete_adapters",
        "get_active_adapters",
    )
    if any(not callable(getattr(pipe, attr, None)) for attr in required):
        raise RuntimeError(
            "The installed Diffusers/Klein pipeline needs load_lora_weights, "
            "set_adapters, delete_adapters and get_active_adapters; upgrade "
            "diffusers/peft or use a dedicated LoRA-capable pipeline."
        )
    if pipe.get_active_adapters():
        raise RuntimeError(
            "The shared Klein pipeline already has active LoRA adapters; "
            "use an unmodified Klein pipeline for object removal."
        )

    # One adapter per call. Clean it up even if denoising fails, so later
    # pool.flux2_klein_pipeline uses are not affected by the removal LoRA.
    adapter_name = "_object_remove_temporary"
    existing = getattr(pipe, "get_list_adapters", lambda: {})()
    if any(adapter_name in names for names in existing.values()):
        raise RuntimeError("Temporary object-removal adapter name is already in use")

    # Download/cache the valid 76 MB checkpoint and validate its contents.
    checkpoint = _validated_object_remove_lora_path(lora_repo, lora_weight)
    try:
        try:
            pipe.load_lora_weights(
                str(checkpoint.parent),
                weight_name=checkpoint.name,
                adapter_name=adapter_name,
            )
        except Exception as exc:
            raise RuntimeError(
                "The 76 MB LoRA checkpoint is valid, but Diffusers failed "
                "to load/convert its diffusion_model.* keys. Use a recent "
                "Diffusers version with Flux2 LoRA conversion support and "
                "the black-forest-labs/FLUX.2-klein-4B base model. "
                f"Checkpoint: {checkpoint}"
            ) from exc
        pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
        config = pool.config.klein
        with torch.inference_mode():
            result = (
                pipe(
                    image=annotated,
                    prompt="Remove the highlighted object from the scene",
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
        # A failing load may have installed a partial adapter already.
        list_adapters = getattr(pipe, "get_list_adapters", None)
        installed = list_adapters() if callable(list_adapters) else {}
        if any(adapter_name in names for names in installed.values()):
            pipe.delete_adapters(adapter_name)

    if result.size != (ww, wh):
        result = result.resize((ww, wh), Image.Resampling.LANCZOS)
    generated_crop = result.resize(original_crop.size, Image.Resampling.LANCZOS)

    # A tiny transition ring avoids an abrupt fish-shaped paste. Pixels in
    # other fish are protected exactly if an explicit protection mask exists.
    if feather_px:
        extended = maximum_filter(
            np.asarray(cropped_mask),
            size=(2 * feather_px + 1, 2 * feather_px + 1),
            mode="constant",
            cval=0,
        )
        softened = Image.fromarray(extended, "L").filter(
            ImageFilter.GaussianBlur(radius=max(1, feather_px / 2))
        )
        blend_mask = ImageChops.lighter(cropped_mask, softened)
    else:
        blend_mask = cropped_mask
    if protection_mask is not None:
        protect_crop = protection_mask.convert("L").crop(crop_box)
        protect_arr = np.asarray(protect_crop)
        protect_threshold = 1 if int(protect_arr.max()) <= 1 else 128
        blend_arr = np.asarray(blend_mask).copy()
        blend_arr[protect_arr >= protect_threshold] = 0
        blend_mask = Image.fromarray(blend_arr, "L")

    output = original.copy()
    output.paste(generated_crop, crop_box[:2], blend_mask)
    return output


__all__ = [
    "flux2_klein_add_object_with_box",
    "flux2_klein_add_object_with_mask",
    "flux2_klein_edit_object",
    "flux2_klein_remove_object",
]
