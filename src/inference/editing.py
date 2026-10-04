from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import maximum_filter

from .inpainting import remove_with_object_clear
from .runtime import ModelCache, device
from utils.image import masked_layer, paste_layer

EditMode = Literal["local", "regional", "global"]


def scale_masked_object(
    models: ModelCache, image: Image.Image, mask: Image.Image, factor: float
) -> Image.Image:
    if factor <= 0:
        raise ValueError("factor must be greater than zero")
    layer, origin = masked_layer(image, mask)
    scaled = layer.resize(
        (max(1, round(layer.width * factor)), max(1, round(layer.height * factor))),
        Image.Resampling.LANCZOS,
    )
    center = (origin[0] + layer.width / 2, origin[1] + layer.height / 2)
    destination = (
        round(center[0] - scaled.width / 2),
        round(center[1] - scaled.height / 2),
    )
    clean = remove_with_object_clear(
        models, image, mask, "remove the selected object", model_key="scale_object"
    )
    return paste_layer(clean, scaled, destination)


class Flux2KleinImageEditBackend:
    """Adapter for FLUX.2 Klein's image-conditioned editing pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        import torch

        result = self.pipeline(
            image=image.convert("RGB"),
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=torch.Generator(device=device()).manual_seed(42),
        )
        return result.images[0].convert("RGB")


def _load_flux2_klein_image_edit_backend(
    model_id: str,
) -> Flux2KleinImageEditBackend:
    from diffusers import Flux2KleinPipeline

    import torch

    model_device = device()
    if model_device.startswith("cuda"):
        dtype = torch.bfloat16
    elif model_device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(model_device)
    return Flux2KleinImageEditBackend(pipeline)


def semantic_edit_crop(
    models: ModelCache, crop: Image.Image, prompt: str
) -> Image.Image:
    model_id = models.config.model_id("edit_object", "black-forest-labs/FLUX.2-klein-4B")
    backend = models.get(
        "flux2_klein_image_edit",
        lambda: _load_flux2_klein_image_edit_backend(model_id),
        model_id=model_id,
    )
    return backend.edit(crop, prompt)


def _editing_prompt(
    prompt: str, mode: EditMode, bounds: tuple[int, int, int, int]
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


def edit_masked_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    mode: EditMode = "local",
    padding: float = 0.25,
) -> Image.Image:
    """Edit one masked object locally, within a padded region, or globally."""
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
    if not isinstance(padding, (int, float)) or not 0 <= padding <= 1:
        raise ValueError("padding must be between 0 and 1")

    mask = mask.convert("L")
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("mask is empty")

    left, top, right, bottom = bounds
    current = image.convert("RGB")
    instruction = _editing_prompt(prompt, mode, bounds)

    if mode == "global":
        edited = semantic_edit_crop(models, current, instruction)
        if edited.size != current.size:
            edited = edited.resize(current.size, Image.Resampling.LANCZOS)
        return edited.convert("RGB")

    if mode == "regional":
        padding_x = round((right - left) * padding)
        padding_y = round((bottom - top) * padding)
        dilated_mask = maximum_filter(
            np.asarray(mask),
            size=(padding_y * 2 + 1, padding_x * 2 + 1),
            mode="constant",
            cval=0,
        )
        allowed_mask = Image.fromarray(dilated_mask)
    else:
        padding_x = max(16, round((right - left) * 0.2))
        padding_y = max(16, round((bottom - top) * 0.2))
        allowed_mask = mask

    crop_box = (
        max(0, left - padding_x),
        max(0, top - padding_y),
        min(image.width, right + padding_x),
        min(image.height, bottom + padding_y),
    )
    original_crop = current.crop(crop_box)
    composite_mask = ImageChops.darker(
        allowed_mask.filter(ImageFilter.GaussianBlur(radius=0.8)), allowed_mask
    )
    edited_crop = semantic_edit_crop(models, original_crop, instruction)
    if edited_crop.size != original_crop.size:
        edited_crop = edited_crop.resize(original_crop.size, Image.Resampling.LANCZOS)

    current.paste(edited_crop, crop_box[:2], composite_mask.crop(crop_box))
    return current
