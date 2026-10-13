"""Load and share the image-editing, segmentation, and object-removal models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from framework.inference.external.objectclear.pipelines.pipeline_objectclear import (
    ObjectClearPipeline,
)
from framework.utils.randomness import seed_random_generators

from .config import InferenceConfig
from .utils import get_device, torch_dtype

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline
    from transformers import (
        GroundingDinoForObjectDetection,
        GroundingDinoProcessor,
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


@dataclass(frozen=True)
class GroundingDinoPipelines:
    model: GroundingDinoForObjectDetection
    processor: GroundingDinoProcessor


class ModelPool:
    """One shared instance for model pipelines and their configurations.

    Public attributes:
        flux2_klein_pipeline          - regular image editing
        sam3_pipelines                - text and point/box segmentation
        grounding_dino_pipelines      - phrase grounding for region segmentation
        object_clear_pipeline         - object removal

    Grounding DINO loads on first use; the other pipelines are initialized eagerly.
    """

    FLUX2_KLEIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
    SAM3_MODEL_ID = "facebook/sam3"
    GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    OBJECT_CLEAR_MODEL_ID = "jixin0101/ObjectClear"

    def __init__(self, config: InferenceConfig | None = None) -> None:
        seed_random_generators()
        self.config = config if config is not None else InferenceConfig()
        self.device = get_device()
        self._pipelines = (
            self.flux2_klein_pipeline,
            self.sam3_pipelines,
            self.object_clear_pipeline,
            self.grounding_dino_pipelines,
        )

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
    def grounding_dino_pipelines(self) -> GroundingDinoPipelines:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        dtype = torch_dtype(self.device)
        return GroundingDinoPipelines(
            model=AutoModelForZeroShotObjectDetection.from_pretrained(
                self.GROUNDING_DINO_MODEL_ID,
                torch_dtype=dtype,
            )
            .to(self.device)
            .eval(),
            processor=AutoProcessor.from_pretrained(self.GROUNDING_DINO_MODEL_ID),
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
