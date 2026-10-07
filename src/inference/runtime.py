from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

_MODEL_ID_KEYS = {
    "edit_object",
    "flip_object",
    "generate_object",
    "generate_object_box",
    "generate_object_gligen",
    "generate_object_mask",
    "move_object",
    "remove_object",
    "replace_object",
    "rotate_object",
    "scale_object",
    "segment_object",
    "shift_object",
}

_BACKEND_OPTIONS = {
    "edit_object": {"flux2-klein-4b", "flux2-klein-9b-kv"},
    "generate_object_box": {"stable-diffusion", "flux-fill"},
    "generate_object_mask": {"stable-diffusion", "flux-fill"},
}


@dataclass(frozen=True)
class ModelConfig:
    """Model IDs and backend choices for image-editing tools."""

    model_ids: Mapping[str, str] = field(default_factory=dict)
    backends: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = self.model_ids.keys() - _MODEL_ID_KEYS
        if unknown:
            raise ValueError(f"unsupported model config keys: {sorted(unknown)}")
        if any(
            not isinstance(model_id, str) or not model_id.strip()
            for model_id in self.model_ids.values()
        ):
            raise ValueError("model IDs must be non-empty strings")
        unknown_backends = self.backends.keys() - _BACKEND_OPTIONS.keys()
        if unknown_backends:
            raise ValueError(
                f"unsupported model backend keys: {sorted(unknown_backends)}"
            )
        invalid_backends = {
            name: backend
            for name, backend in self.backends.items()
            if not isinstance(backend, str) or backend not in _BACKEND_OPTIONS[name]
        }
        if invalid_backends:
            choices = {
                name: sorted(_BACKEND_OPTIONS[name]) for name in invalid_backends
            }
            raise ValueError(
                f"unsupported model backend choices {invalid_backends}; "
                f"expected one of {choices}"
            )
        object.__setattr__(self, "model_ids", MappingProxyType(dict(self.model_ids)))
        object.__setattr__(self, "backends", MappingProxyType(dict(self.backends)))

    def model_id(self, tool_name: str, default: str) -> str:
        if tool_name in ("generate_object_box", "generate_object_mask"):
            return self.model_ids.get(
                tool_name, self.model_ids.get("generate_object", default)
            )
        return self.model_ids.get(tool_name, default)

    def backend(self, tool_name: str, default: str) -> str:
        return self.backends.get(tool_name, default)


class ModelCache:
    """Share lazily loaded model pipelines across tools in one agent."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config if config is not None else ModelConfig()
        self._models: dict[tuple[str, str | None], Any] = {}

    def get(
        self, name: str, loader: Callable[[], Any], *, model_id: str | None = None
    ) -> Any:
        key = (name, model_id)
        if key not in self._models:
            self._models[key] = loader()
        return self._models[key]

    def close(self) -> None:
        self._models.clear()


def device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(device_name: str):
    import torch

    return torch.float16 if device_name.startswith("cuda") else torch.float32
