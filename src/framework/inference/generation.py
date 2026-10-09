from __future__ import annotations

from PIL import Image

from .runtime import ModelCache, device, torch_dtype


def _load_flux():
    from diffusers import FluxPipeline

    model_device = device()
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch_dtype(model_device),
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).to(model_device)
    if model_device.startswith("cuda"):
        pipeline.enable_attention_slicing(slice_size="auto")
    return pipeline


def generate_with_flux(models: ModelCache, prompt: str) -> Image.Image:
    pipeline = models.get("flux", _load_flux)
    return pipeline(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=512,
        width=512,
    ).images[0].convert("RGB")
