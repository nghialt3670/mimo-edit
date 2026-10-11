from __future__ import annotations

from dataclasses import dataclass

from framework.inference import ModelPool


@dataclass(frozen=True)
class ToolContext:
    model_pool: ModelPool
