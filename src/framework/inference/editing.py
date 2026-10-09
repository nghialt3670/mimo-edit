from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import maximum_filter

from .inpainting import remove_with_object_clear
from .runtime import ModelCache, device
from framework.utils.image import masked_layer, paste_layer

EditMode = Literal["local", "regional", "global"]

FLUX2_KLEIN_MODEL_IDS = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-9b-kv": "black-forest-labs/FLUX.2-klein-9b-kv",
}


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

    def __init__(self, pipeline: Any, guidance_scale: float | None = 1.0):
        self.pipeline = pipeline
        self.guidance_scale = guidance_scale

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        import torch

        kwargs: dict[str, Any] = {
            "image": image.convert("RGB"),
            "prompt": prompt,
            "num_inference_steps": 4,
            "generator": torch.Generator(device=device()).manual_seed(42),
        }
        if self.guidance_scale is not None:
            kwargs["guidance_scale"] = self.guidance_scale
        result = self.pipeline(
            **kwargs
        )
        return result.images[0].convert("RGB")


def _load_flux2_klein_image_edit_backend(
    model_id: str,
    backend: str = "flux2-klein-4b",
) -> Flux2KleinImageEditBackend:
    if backend == "flux2-klein-4b":
        from diffusers import Flux2KleinPipeline as Pipeline
    elif backend == "flux2-klein-9b-kv":
        from diffusers import Flux2KleinKVPipeline as Pipeline
    else:
        raise ValueError(f"unsupported FLUX.2 Klein backend: {backend}")

    import torch

    model_device = device()
    if model_device.startswith("cuda"):
        dtype = torch.bfloat16
    elif model_device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    pipeline = Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(model_device)
    guidance_scale = 1.0 if backend == "flux2-klein-4b" else None
    return Flux2KleinImageEditBackend(pipeline, guidance_scale=guidance_scale)


def semantic_edit_crop(
    models: ModelCache, crop: Image.Image, prompt: str
) -> Image.Image:
    backend_name = models.config.backend("edit_object", "flux2-klein-4b")
    default_model_id = FLUX2_KLEIN_MODEL_IDS[backend_name]
    model_id = models.config.model_id("edit_object", default_model_id)
    backend = models.get(
        backend_name,
        lambda: _load_flux2_klein_image_edit_backend(model_id, backend_name),
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
    """Edit one masked object and feather the local/regional composite boundary."""
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
    edited_crop = semantic_edit_crop(models, original_crop, instruction)
    if edited_crop.size != original_crop.size:
        edited_crop = edited_crop.resize(original_crop.size, Image.Resampling.LANCZOS)

    current.paste(edited_crop, crop_box[:2], composite_mask.crop(crop_box))
    return current
