"""MICE bbox editing helpers for FLUX.2 Klein.

Place this file at ``src/framework/inference/external/mice_adapter.py``.
Clone https://github.com/Blowing-Up-Groundhogs/mice.git next to it into
``src/framework/inference/external/mice_upstream``.

This uses the AUTHORS' custom FLUX.2 Klein pipeline, not the regular
``diffusers.Flux2KleinPipeline``.  MICE's Apple MPS support is experimental.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from typing import Any


MICE_SOURCE = Path(__file__).resolve().parent / "mice_upstream"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def _mice_imports() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import custom upstream classes lazily to avoid affecting other inference modules."""
    if not (MICE_SOURCE / "flux2" / "pipeline_flux2_klein.py").is_file():
        raise FileNotFoundError(
            f"MICE source not found in {MICE_SOURCE}.\n"
            "Clone it there with:\n"
            f"git clone https://github.com/Blowing-Up-Groundhogs/mice.git {MICE_SOURCE}"
        )
    if str(MICE_SOURCE) not in sys.path:
        sys.path.insert(0, str(MICE_SOURCE))

    from flux2.attention import AttentionSetting, get_attention_processors
    from flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from flux2.transformer_flux2_klein import (
        Flux2Attention,
        Flux2ParallelSelfAttention,
        Flux2Transformer2DModel,
    )

    return (
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
        Flux2Attention,
        Flux2ParallelSelfAttention,
        AttentionSetting,
        get_attention_processors,
    )


def load_mice_pipeline(
    *,
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    strict: bool = False,
    kernel_size: int = 11,
    temperature: float = 3.0,
    sigma_scale: float = 0.6,
):
    """Load and configure MICE once; reuse its returned pipeline for edits.

    The upstream release was written/tested for CUDA + bfloat16.  MPS defaults
    to float32 here to reduce dtype problems, but is *not* verified upstream.
    """
    (
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
        Flux2Attention,
        Flux2ParallelSelfAttention,
        AttentionSetting,
        get_attention_processors,
    ) = _mice_imports()

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    setting = "apitasmkernelnonlapstrict" if strict else "apitasmkernelnonlap"
    processors = get_attention_processors(
        AttentionSetting(setting), sigma_scale, kernel_size, temperature
    )
    if isinstance(processors, tuple) and len(processors) == 2:
        regular, parallel = processors
        for module in pipe.transformer.modules():
            if isinstance(module, Flux2Attention):
                module.set_processor(regular)
            elif isinstance(module, Flux2ParallelSelfAttention):
                module.set_processor(parallel)
        pipe._mice_processors = (regular, parallel)
    else:
        pipe.transformer.set_attn_processor(processors)
        pipe._mice_processors = (processors,)

    pipe._mice_attention_setting = setting
    return pipe


@torch.inference_mode()
def mice_add_object_with_box(
    pipe,
    image: Image.Image,
    box: Sequence[float],
    prompt: str,
    *,
    steps: int = 4,
    seed: int = 0,
    guidance_scale: float = 1.0,
    prompt_settings: str = "outer_local_prompts",
    preserve_outside_box: bool = False,
    max_side: int = 1024,
) -> Image.Image:
    """Edit/add in a pixel-space XYXY box using MICE's spatial attention.

    This is not guaranteed to place a new object correctly or preserve pixels
    outside the box unless ``preserve_outside_box=True`` (hard final paste).
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
        raise ValueError("box must be four finite XYXY pixel coordinates")
    if steps < 1:
        raise ValueError("steps must be positive")
    if max_side < 64:
        raise ValueError("max_side must be at least 64")

    original = image.convert("RGB")
    width, height = original.size
    x1, y1, x2, y2 = map(float, box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box must be inside image and have positive width/height")
    norm_box = [x1 / width, y1 / height, x2 / width, y2 / height]

    # Respect common FLUX image-size requirements while keeping the box normalized.
    scale = min(1.0, max_side / max(width, height))
    work_width = max(64, math.ceil(width * scale / 16) * 16)
    work_height = max(64, math.ceil(height * scale / 16) * 16)
    working = (
        original.resize((work_width, work_height), Image.Resampling.LANCZOS)
        if (work_width, work_height) != original.size
        else original
    )

    # Upstream attention processors cache masks; clear between images/boxes.
    processors = getattr(pipe, "_mice_processors", ())
    if not processors:
        raise ValueError(
            "Expected a MICE-configured pipeline. Use load_mice_pipeline() "
            "rather than the regular Diffusers FLUX.2 Klein pipeline."
        )
    for proc in processors:
        if hasattr(proc, "clear_cached_masks"):
            proc.clear_cached_masks()

    device = str(pipe._execution_device)
    # torch.Generator(device='mps') is not portable across PyTorch builds.
    generator_device = "cpu" if device.startswith("mps") else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    result = pipe(
        image=working,
        prompt=prompt.strip(),
        height=work_height,
        width=work_width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        prompt_settings=prompt_settings,
        attention_setting=pipe._mice_attention_setting,
        hard_image_attribute_binding_list_double=list(range(0, 5)),
        hard_image_attribute_binding_list_single=list(range(0, 20)),
        bring_area_to_1024_squared=False,
        generator=generator,
        hard_masking_steps=list(range(steps)),
        relaxed_timesteps="soft",
        attention_kwargs={"smooth_P_L": False},
        free_latent=False,
        free_context=False,
        free_LC=False,
        instance_bboxes_xyxy_normalized=[norm_box],
    )

    edited = result.images[0].convert("RGB")
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    if preserve_outside_box:
        mask = Image.new("L", original.size, 0)
        ImageDraw.Draw(mask).rectangle(
            (math.floor(x1), math.floor(y1), math.ceil(x2) - 1, math.ceil(y2) - 1),
            fill=255,
        )
        return Image.composite(edited, original, mask)
    return edited


__all__ = ["load_mice_pipeline", "mice_add_object_with_box"]
