from __future__ import annotations

from PIL import Image

from .runtime import ModelCache, device, torch_dtype
from utils.image import validate_box


def _load_object_clear(model_id: str):
    from .objectclear.pipelines import ObjectClearPipeline

    model_device = device()
    pipeline = ObjectClearPipeline.from_pretrained_with_custom_modules(
        model_id,
        torch_dtype=torch_dtype(model_device),
        apply_attention_guided_fusion=True,
        variant="fp16" if model_device.startswith("cuda") else None,
        low_cpu_mem_usage=True,
    ).to(model_device)
    if model_device.startswith("cuda"):
        pipeline.enable_attention_slicing(slice_size="auto")
    return pipeline


def remove_with_object_clear(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    model_key: str = "remove_object",
) -> Image.Image:
    import torch

    model_id = models.config.model_id(model_key, "jixin0101/ObjectClear")
    pipeline = models.get(
        "object_clear", lambda: _load_object_clear(model_id), model_id=model_id
    )
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    original_size = image.size
    scale = 384 / min(image.size)
    working_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    result = pipeline(
        prompt=prompt or "remove the instance of object",
        image=image.resize(working_size, Image.Resampling.BILINEAR),
        mask_image=mask.resize(working_size, Image.Resampling.NEAREST),
        generator=torch.Generator(device=pipeline.unet.device).manual_seed(42),
        num_inference_steps=8,
        guidance_scale=2.0,
        height=working_size[1],
        width=working_size[0],
        return_attn_map=False,
    )
    return result.images[0].resize(original_size, Image.Resampling.BICUBIC).convert("RGB")


def _load_stable_diffusion(model_id: str):
    from diffusers import StableDiffusionInpaintPipeline

    model_device = device()
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype(model_device),
        variant="fp16" if model_device.startswith("cuda") else None,
        low_cpu_mem_usage=True,
    ).to(model_device)
    if model_device.startswith("cuda"):
        pipeline.enable_attention_slicing(slice_size="auto")
    return pipeline


def _load_flux_fill(model_id: str):
    from diffusers import FluxFillPipeline

    import torch

    model_device = device()
    if model_device.startswith("cuda"):
        dtype = torch.bfloat16
    elif model_device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return FluxFillPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(model_device)


def _match_image_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _inpaint_with_flux_fill(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    model_key: str,
) -> Image.Image:
    import torch

    original_size = image.size
    default_model_id = "black-forest-labs/FLUX.1-Fill-dev"
    model_id = models.config.model_id(model_key, default_model_id)
    pipeline = models.get(
        "flux_fill",
        lambda: _load_flux_fill(model_id),
        model_id=model_id,
    )
    width = max(16, round(image.width / 16) * 16)
    height = max(16, round(image.height / 16) * 16)
    image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    mask = mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
    result = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=height,
        width=width,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator(device=device()).manual_seed(42),
    )
    return _match_image_size(result.images[0], original_size)


def inpaint_with_stable_diffusion(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    model_key: str = "generate_object",
) -> Image.Image:
    import torch

    original_size = image.size
    default_model = (
        "stable-diffusion-v1-5/stable-diffusion-inpainting"
        if model_key == "edit_object"
        else "runwayml/stable-diffusion-inpainting"
    )
    model_id = models.config.model_id(model_key, default_model)
    pipeline = models.get(
        "stable_diffusion",
        lambda: _load_stable_diffusion(model_id),
        model_id=model_id,
    )
    result = pipeline(
        prompt=prompt,
        image=image.convert("RGB"),
        mask_image=mask.convert("L").resize(image.size, Image.Resampling.NEAREST),
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.Generator(device=device()).manual_seed(42),
    )
    return _match_image_size(result.images[0], original_size)


def inpaint_with_configured_backend(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    route_key: str,
) -> Image.Image:
    if route_key not in ("generate_object_mask", "generate_object_box"):
        raise ValueError(f"unsupported generate_object route: {route_key}")
    selected_backend = models.config.backend(route_key, "stable-diffusion")
    if selected_backend == "flux-fill":
        return _inpaint_with_flux_fill(
            models, image, mask, prompt, model_key=route_key
        )
    if selected_backend == "stable-diffusion":
        return inpaint_with_stable_diffusion(
            models, image, mask, prompt, model_key=route_key
        )
    raise ValueError(f"unsupported generate_object backend: {selected_backend}")


def _load_gligen(model_id: str):
    from diffusers import StableDiffusionGLIGENPipeline

    model_device = device()
    kwargs = {"torch_dtype": torch_dtype(model_device), "low_cpu_mem_usage": True}
    if model_device.startswith("cuda"):
        kwargs["variant"] = "fp16"
    return StableDiffusionGLIGENPipeline.from_pretrained(model_id, **kwargs).to(
        model_device
    )


def inpaint_with_gligen(
    models: ModelCache,
    image: Image.Image,
    prompt: str,
    phrases: list[str],
    boxes: list[list[float]],
    *,
    model_key: str = "generate_object_gligen",
) -> Image.Image:
    import torch

    original_size = image.size
    if len(phrases) != len(boxes):
        raise ValueError("phrases and boxes must have the same length")
    normalized_boxes = []
    for values in boxes:
        x1, y1, x2, y2 = validate_box(values)
        normalized_boxes.append(
            [
                max(0.0, min(1.0, x1 / image.width)),
                max(0.0, min(1.0, y1 / image.height)),
                max(0.0, min(1.0, x2 / image.width)),
                max(0.0, min(1.0, y2 / image.height)),
            ]
        )
    model_id = models.config.model_id(
        model_key, "masterful/gligen-1-4-inpainting-text-box"
    )
    pipeline = models.get(
        "gligen", lambda: _load_gligen(model_id), model_id=model_id
    )
    result = pipeline(
        prompt=prompt,
        gligen_phrases=phrases,
        gligen_boxes=normalized_boxes,
        gligen_inpaint_image=image.convert("RGB"),
        gligen_scheduled_sampling_beta=0.3,
        num_inference_steps=50,
        output_type="pil",
        generator=torch.Generator(device=device()).manual_seed(42),
    )
    return _match_image_size(result.images[0], original_size)
