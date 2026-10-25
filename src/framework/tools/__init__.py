from __future__ import annotations

from typing import Literal

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
from .registry import ToolRegistry, build_tools, register_tool, registry, text_mode_registry


def image_editing_tools(
    pool: ModelPool | None = None,
    *,
    inference_config: InferenceConfig | None = None,
    prompt_mode: Literal["mixed", "text"] = "mixed",
) -> list[Tool]:
    """Build tools sharing one eagerly initialized inference model pool.

    In "text" mode, `segment_object` is replaced by a text-only variant.
    """
    if pool is not None and inference_config is not None:
        raise ValueError("pass either pool or inference_config, not both")
    if prompt_mode not in ("mixed", "text"):
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")
    model_pool = pool if pool is not None else ModelPool(inference_config)
    context = ToolContext(model_pool=model_pool)
    if prompt_mode == "mixed":
        return build_tools(context)
    tools = [tool for tool in build_tools(context) if tool.name != "segment_object"]
    return tools + text_mode_registry.build(context)


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
