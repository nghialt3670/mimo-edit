from __future__ import annotations

from PIL import Image

from .runtime import ModelCache, device, torch_dtype

DEFAULT_OBJECT_CLEAR_MODEL_ID = "jixin0101/ObjectClear"


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

    model_id = models.config.model_id(model_key, DEFAULT_OBJECT_CLEAR_MODEL_ID)
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
