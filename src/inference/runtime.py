from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

_MODEL_ID_KEYS = {
    "edit_object",
    "flip_object",
    "generate_object",
    "generate_object_gligen",
    "move_object",
    "remove_object",
    "replace_object",
    "rotate_object",
    "scale_object",
    "segment_object",
    "shift_object",
}


@dataclass(frozen=True)
class ModelConfig:
    """Model choices for tools; IDs are keyed by tool name."""

    model_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = self.model_ids.keys() - _MODEL_ID_KEYS
        if unknown:
            raise ValueError(f"unsupported model config keys: {sorted(unknown)}")
        if any(
            not isinstance(model_id, str) or not model_id.strip()
            for model_id in self.model_ids.values()
        ):
            raise ValueError("model IDs must be non-empty strings")
        object.__setattr__(self, "model_ids", MappingProxyType(dict(self.model_ids)))

    def model_id(self, tool_name: str, default: str) -> str:
        return self.model_ids.get(tool_name, default)


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
