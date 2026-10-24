from __future__ import annotations

from smolagents import Tool

from framework.inference import (
    InferenceConfig,
    KleinInferenceConfig,
    ModelPool,
    ObjectClearInferenceConfig,
    QwenLocateInferenceConfig,
    Sam3InferenceConfig,
)

from . import image_editing as image_editing
from .context import ToolContext
from .registry import ToolRegistry, build_tools, register_tool, registry


def image_editing_tools(
    pool: ModelPool | None = None,
    *,
    inference_config: InferenceConfig | None = None,
) -> list[Tool]:
    """Build tools sharing one eagerly initialized inference model pool."""
    if pool is not None and inference_config is not None:
        raise ValueError("pass either pool or inference_config, not both")
    model_pool = pool if pool is not None else ModelPool(inference_config)
    return build_tools(ToolContext(model_pool=model_pool))


__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolContext",
    "ModelPool",
    "InferenceConfig",
    "KleinInferenceConfig",
    "ObjectClearInferenceConfig",
    "QwenLocateInferenceConfig",
    "Sam3InferenceConfig",
    "build_tools",
    "image_editing_tools",
    "register_tool",
    "registry",
]
