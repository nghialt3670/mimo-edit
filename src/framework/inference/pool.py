"""Load and share the image-editing, segmentation, and object-removal models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from framework.inference.external.objectclear.pipelines.pipeline_objectclear import (
    ObjectClearPipeline,
)

from .config import InferenceConfig
from .utils import get_device, torch_dtype

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline
    from transformers import (
        Sam3Model,
        Sam3Processor,
        Sam3TrackerModel,
        Sam3TrackerProcessor,
    )

    from .external.objectclear.pipelines.pipeline_objectclear import ObjectClearPipeline


@dataclass(frozen=True)
class Sam3Pipelines:
    text_model: Sam3Model
    text_processor: Sam3Processor
    tracker_model: Sam3TrackerModel
    tracker_processor: Sam3TrackerProcessor


class ModelPool:
    """One shared instance with all model pipelines loaded during initialization.

    Public attributes:
        flux2_klein_pipeline          - regular image editing
        flux2_klein_inpaint_pipeline  - masked editing (shares Klein weights)
        sam3                          - text and point/box segmentation
        object_clear                  - object removal
    """

    FLUX2_KLEIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
    SAM3_MODEL_ID = "facebook/sam3"
    OBJECT_CLEAR_MODEL_ID = "jixin0101/ObjectClear"

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config if config is not None else InferenceConfig()
        self.device = get_device()
        self._pipelines = (
            self.flux2_klein_pipeline,
            self.sam3_pipelines,
            self.object_clear_pipeline,
        )

    @property
    def sam3(self) -> Sam3Pipelines:
        return self.sam3_pipelines

    @property
    def object_clear(self) -> ObjectClearPipeline:
        return self.object_clear_pipeline

    @cached_property
    def flux2_klein_pipeline(self) -> Flux2KleinPipeline:
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(
            self.FLUX2_KLEIN_MODEL_ID,
            torch_dtype=torch_dtype(
                self.device, cuda_dtype=torch.bfloat16, mps_dtype=torch.float16
            ),
            low_cpu_mem_usage=True,
        )
        # The 4-step Klein 4B checkpoint is distilled (no classifier-free guidance).
        pipe.register_to_config(is_distilled=True)
        return pipe.to(self.device)

    @cached_property
    def sam3_pipelines(self) -> Sam3Pipelines:
        from transformers import (
            Sam3Model,
            Sam3Processor,
            Sam3TrackerModel,
            Sam3TrackerProcessor,
        )

        dtype = torch_dtype(self.device)
        return Sam3Pipelines(
            text_model=Sam3Model.from_pretrained(self.SAM3_MODEL_ID, torch_dtype=dtype)
            .to(self.device)
            .eval(),
            text_processor=Sam3Processor.from_pretrained(self.SAM3_MODEL_ID),
            tracker_model=Sam3TrackerModel.from_pretrained(
                self.SAM3_MODEL_ID, torch_dtype=dtype
            )
            .to(self.device)
            .eval(),
            tracker_processor=Sam3TrackerProcessor.from_pretrained(self.SAM3_MODEL_ID),
        )

    @cached_property
    def object_clear_pipeline(self) -> ObjectClearPipeline:
        from .external.objectclear.pipelines import ObjectClearPipeline

        pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            self.OBJECT_CLEAR_MODEL_ID,
            torch_dtype=torch_dtype(self.device),
            apply_attention_guided_fusion=True,
            variant="fp16" if self.device.startswith("cuda") else None,
            low_cpu_mem_usage=True,
        ).to(self.device)
        if self.device.startswith("cuda"):
            pipe.enable_attention_slicing(slice_size="auto")
        return pipe
