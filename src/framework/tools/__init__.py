from __future__ import annotations

from smolagents import Tool

from framework.inference import ModelCache, ModelConfig
from . import image_editing as image_editing
from .registry import ToolRegistry, build_tools, register_tool, registry


def image_editing_tools(
    model_cache: ModelCache | None = None,
    *,
    model_config: ModelConfig | None = None,
) -> list[Tool]:
    """Build tools sharing a lazy model cache with optional model overrides."""
    if model_cache is not None and model_config is not None:
        raise ValueError("pass either model_cache or model_config, not both")
    cache = model_cache if model_cache is not None else ModelCache(model_config)
    return build_tools(cache)


__all__ = [
    "Tool",
    "ToolRegistry",
    "ModelCache",
    "ModelConfig",
    "build_tools",
    "image_editing_tools",
    "register_tool",
    "registry",
]