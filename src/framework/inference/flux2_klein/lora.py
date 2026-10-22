"""Load and switch FLUX.2 Klein LoRA adapters on the shared pipeline.

Diffusers/PEFT keep several adapters on one pipeline. Load each once under a
stable ``adapter_name``, then ``set_adapters`` to the one needed for this call.
``unload_lora_weights`` drops every adapter, so edit and remove must never
unload after a generation; ``disable_lora`` is enough to restore the base model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

import torch
from PIL import Image

from framework.utils.randomness import get_random_seed

from ..config import KleinInferenceConfig

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline

RemoveLoraId = Literal["fal", "slopshop"]

# Edit-consistency LoRA (FLUX.2 Klein 4B, not 9B).
CONSISTENCY_LORA_REPO = "lrzjason/Consistance_Edit_Lora"
CONSISTENCY_LORA_FILE = "f2k_4B_consist_20260314.safetensors"
CONSISTENCY_ADAPTER_NAME = "klein_4b_consistency"

OBJECT_REMOVE_LORA_REPO = "fal/flux-2-klein-4B-object-remove-lora"
# The 58 MB flux-object-remove-lora.safetensors upload is incomplete upstream.
# Use the validated, 76 MB ComfyUI-converted checkpoint instead. Diffusers'
# Flux2 LoRA loader converts its diffusion_model.* keys automatically.
OBJECT_REMOVE_LORA_WEIGHT = (
    "kDEkt5q7tDLKOpQJIVMPx_" "pytorch_lora_weights_comfy_converted.safetensors"
)
OBJECT_REMOVE_ADAPTER_NAME = "klein_4b_object_remove"

SLOPSHOP_ERASE_REPO = "slopshop/erase-v1"
SLOPSHOP_ERASE_WEIGHT = "erase_v1_diffusers.safetensors"
SLOPSHOP_ERASE_EMBEDS = "prompt_embeds.safetensors"
SLOPSHOP_ERASE_ADAPTER_NAME = "klein_4b_slopshop_erase"
SLOPSHOP_ERASE_PROMPT = (
    "Remove the object under the white area of the mask (second image), "
    "including its shadow and reflection, and fill that area naturally with "
    "the surrounding background."
)


@dataclass(frozen=True)
class RemoveLoraSpec:
    """How to load and feed one object-removal LoRA."""

    adapter_name: str
    repo: str
    weight: str
    scale: float
    prompt: str
    highlight: bool
    validate_comfy_keys: bool = False
    mask_grow_frac: float = 0.0
    prompt_embeds_file: str | None = None


REMOVE_LORAS: dict[RemoveLoraId, RemoveLoraSpec] = {
    "fal": RemoveLoraSpec(
        adapter_name=OBJECT_REMOVE_ADAPTER_NAME,
        repo=OBJECT_REMOVE_LORA_REPO,
        weight=OBJECT_REMOVE_LORA_WEIGHT,
        scale=1.3,
        prompt="Remove the highlighted object from the scene",
        highlight=True,
        validate_comfy_keys=True,
    ),
    "slopshop": RemoveLoraSpec(
        adapter_name=SLOPSHOP_ERASE_ADAPTER_NAME,
        repo=SLOPSHOP_ERASE_REPO,
        weight=SLOPSHOP_ERASE_WEIGHT,
        scale=1.0,
        prompt=SLOPSHOP_ERASE_PROMPT,
        highlight=False,
        mask_grow_frac=0.02,
        prompt_embeds_file=SLOPSHOP_ERASE_EMBEDS,
    ),
}

_PROMPT_EMBEDS: dict[str, torch.Tensor] = {}

# The model pool may reuse a Klein pipeline across operations/threads.
_KLEIN_INFERENCE_LOCK = RLock()
_LORA_SOURCES_ATTR = "_mimo_klein_lora_sources"

_REQUIRED_LORA_API = (
    "load_lora_weights",
    "set_adapters",
    "disable_lora",
    "get_list_adapters",
)


def installed_adapter_names(pipeline: Flux2KleinPipeline) -> set[str]:
    adapters = pipeline.get_list_adapters()
    names: set[str] = set()
    for value in adapters.values():
        names.update(value)
    return names


def require_lora_api(pipeline: Flux2KleinPipeline) -> None:
    missing = [
        name
        for name in _REQUIRED_LORA_API
        if not callable(getattr(pipeline, name, None))
    ]
    if missing:
        raise RuntimeError(
            "The Klein pipeline needs load_lora_weights, set_adapters, "
            "disable_lora and get_list_adapters (recent diffusers + peft). "
            f"Missing: {', '.join(missing)}"
        )


def remove_lora_spec(name: str) -> RemoveLoraSpec:
    if name not in REMOVE_LORAS:
        choices = ", ".join(sorted(REMOVE_LORAS))
        raise ValueError(f"remove_lora must be one of: {choices}")
    return REMOVE_LORAS[name]


def validated_object_remove_lora_path(
    repo_or_path: str,
    weight_name: str,
    *,
    validate_comfy_keys: bool = False,
) -> Path:
    """Resolve a Hub or local checkpoint and reject unreadable safetensors.

    Do not retry a malformed fal checkpoint: the original 58 MB upstream
    upload is known to have missing tensor data and redownloading cannot
    repair it.
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
            if validate_comfy_keys and checkpoint.name == OBJECT_REMOVE_LORA_WEIGHT:
                if not all(key.startswith("diffusion_model.") for key in keys):
                    raise ValueError("unexpected keys in the 76 MB object-removal LoRA")
    except Exception as exc:
        raise RuntimeError(
            f"Unreadable object-removal LoRA checkpoint: {checkpoint}. "
            "The original 58 MB fal checkpoint is incomplete upstream. "
            f"Use {OBJECT_REMOVE_LORA_WEIGHT} from {OBJECT_REMOVE_LORA_REPO} "
            f"or {SLOPSHOP_ERASE_WEIGHT} from {SLOPSHOP_ERASE_REPO}."
        ) from exc

    return checkpoint


def load_remove_prompt_embeds(spec: RemoveLoraSpec) -> torch.Tensor | None:
    """Return the LoRA's precomputed prompt tensor, if it ships one."""
    if not spec.prompt_embeds_file:
        return None
    cached = _PROMPT_EMBEDS.get(spec.adapter_name)
    if cached is not None:
        return cached
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(repo_id=spec.repo, filename=spec.prompt_embeds_file)
    embeds = load_file(path)["prompt_embeds"]
    _PROMPT_EMBEDS[spec.adapter_name] = embeds
    return embeds


def ensure_adapter(
    pipeline: Flux2KleinPipeline,
    adapter_name: str,
    pretrained: str,
    weight_name: str,
) -> None:
    """Load ``adapter_name`` once. A second LoRA uses a different name, not unload."""
    require_lora_api(pipeline)
    sources: dict[str, tuple[str, str]] = getattr(pipeline, _LORA_SOURCES_ATTR, {})
    source = (pretrained, weight_name)
    if adapter_name in installed_adapter_names(pipeline):
        previous = sources.get(adapter_name)
        if previous is not None and previous != source:
            raise RuntimeError(
                f"LoRA adapter {adapter_name!r} is already loaded from "
                f"{previous[0]}/{previous[1]}, not {pretrained}/{weight_name}"
            )
        sources[adapter_name] = source
        setattr(pipeline, _LORA_SOURCES_ATTR, sources)
        return

    try:
        pipeline.load_lora_weights(
            pretrained, weight_name=weight_name, adapter_name=adapter_name
        )
    except Exception as exc:
        raise RuntimeError(
            f"Diffusers failed to load LoRA {adapter_name!r}: "
            f"{pretrained}/{weight_name}"
        ) from exc
    if adapter_name not in installed_adapter_names(pipeline):
        raise RuntimeError(
            f"LoRA {adapter_name!r} did not register with the Klein pipeline; "
            "check Diffusers/PEFT compatibility and the LoRA weight format."
        )
    sources = dict(getattr(pipeline, _LORA_SOURCES_ATTR, {}))
    sources[adapter_name] = source
    setattr(pipeline, _LORA_SOURCES_ATTR, sources)


def _align_tensor(pipeline: Flux2KleinPipeline, tensor: torch.Tensor) -> torch.Tensor:
    """Copy ``tensor`` onto the pipeline device and transformer dtype."""
    transformer = getattr(pipeline, "transformer", None)
    device = getattr(pipeline, "device", None) or getattr(
        transformer, "device", tensor.device
    )
    dtype = getattr(transformer, "dtype", tensor.dtype)
    return tensor.to(device=device, dtype=dtype)


def generate_klein(
    pipeline: Flux2KleinPipeline,
    image: Image.Image | list[Image.Image],
    prompt: str | None,
    config: KleinInferenceConfig,
    *,
    prompt_embeds: torch.Tensor | None = None,
    adapter_name: str | None = None,
    adapter_weight: float = 0.0,
    pretrained: str | None = None,
    weight_name: str | None = None,
    seed: int | None = None,
    seed_offset: int = 0,
    height: int | None = None,
    width: int | None = None,
) -> Image.Image:
    """Run Klein with at most one named adapter active.

    ``set_adapters([name], adapter_weights=[scale])`` selects that adapter
    without merging any other loaded LoRA. After generation, ``disable_lora``
    returns to the base weights; loaded adapters stay on the pipeline.
    """
    if prompt is not None and prompt_embeds is not None:
        raise ValueError("pass prompt or prompt_embeds, not both")
    if prompt is None and prompt_embeds is None:
        raise ValueError("prompt or prompt_embeds is required")
    use_lora = bool(adapter_name) and adapter_weight > 0
    if use_lora and (not adapter_name or not pretrained or not weight_name):
        raise ValueError("pretrained and weight_name are required to enable a LoRA")

    with _KLEIN_INFERENCE_LOCK:
        if use_lora:
            ensure_adapter(pipeline, adapter_name, pretrained, weight_name)
            enable_lora = getattr(pipeline, "enable_lora", None)
            if callable(enable_lora):
                enable_lora()
            # A one-element list is an exclusive switch, not a merge of both LoRAs.
            pipeline.set_adapters([adapter_name], adapter_weights=[adapter_weight])
        else:
            disable = getattr(pipeline, "disable_lora", None)
            if callable(disable):
                disable()

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            kwargs: dict = {
                "image": image,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "generator": torch.Generator(device="cpu").manual_seed(
                    (get_random_seed() if seed is None else seed) + seed_offset
                ),
            }
            if prompt_embeds is not None:
                # safetensors loads onto CPU; Flux2KleinPipeline.encode_prompt
                # does not move caller-supplied embeds onto the pipeline device.
                kwargs["prompt_embeds"] = _align_tensor(pipeline, prompt_embeds)
            else:
                kwargs["prompt"] = prompt
            if height is not None:
                kwargs["height"] = height
            if width is not None:
                kwargs["width"] = width
            with torch.inference_mode():
                return pipeline(**kwargs).images[0].convert("RGB")
        finally:
            disable = getattr(pipeline, "disable_lora", None)
            if callable(disable):
                disable()
