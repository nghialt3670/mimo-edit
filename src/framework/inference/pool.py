"""Load and share the image-editing, segmentation, and object-removal models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch

from framework.utils.randomness import seed_random_generators

from .config import InferenceConfig
from .utils import get_device, torch_dtype

if TYPE_CHECKING:
    from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
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


@dataclass(frozen=True)
class QwenLocatePipelines:
    model: Any
    processor: Any


class ModelPool:
    """Lazily load and cache model pipelines.

    ``flux2_klein_inpaint_pipeline`` and ``flux2_klein_pipeline`` share their
    underlying model components. Do not call them concurrently or move one
    of the pipelines to another dtype/device without considering both.

    Public attributes:
        flux2_klein_pipeline         - ordinary Klein image editing
        flux2_klein_inpaint_pipeline - masked Klein inpainting / object removal
        mice_pipeline                - MICE spatially constrained Klein editing
        sam3_pipelines               - text and point/box segmentation
        qwen_locate_pipelines        - region bounding-box location
        object_clear_pipeline        - independent ObjectClear object removal
        pctnet_pipeline              - optional color harmonization
    """

    FLUX2_KLEIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
    SAM3_MODEL_ID = "facebook/sam3"
    QWEN_LOCATE_MODEL_ID = "Qwen/Qwen3.5-2B"
    OBJECT_CLEAR_MODEL_ID = "jixin0101/ObjectClear"
    VITMATTE_MODEL_ID = "hustvl/vitmatte-base-composition-1k"

    def __init__(self, config: InferenceConfig | None = None) -> None:
        seed_random_generators()
        self.config = config if config is not None else InferenceConfig()
        self.device = get_device()
        self._pipelines = {
            self.flux2_klein_pipeline,
            self.flux2_klein_inpaint_pipeline,
            self.sam3_pipelines,
        }

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
        # black-forest-labs/FLUX.2-klein-4B is the four-step distilled model.
        pipe.register_to_config(is_distilled=True)
        return pipe.to(self.device)

    @cached_property
    def flux2_klein_inpaint_pipeline(self) -> Flux2KleinInpaintPipeline:
        """Create an inpaint pipeline without downloading a second Klein model.

        Use the concrete class, not AutoPipelineForInpainting: current Diffusers
        AutoPipeline's inpaint mapping does not include FLUX.2 Klein.
        ``from_pipe`` shares the already-loaded weights and only constructs a
        new pipeline wrapper (including its mask processor).
        """
        try:
            from diffusers import Flux2KleinInpaintPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Your Diffusers installation does not provide "
                "Flux2KleinInpaintPipeline. Install a version that contains "
                "this class (the current Diffusers source branch does)."
            ) from exc

        editing_pipe = self.flux2_klein_pipeline
        model_dtype = editing_pipe.transformer.dtype
        is_distilled = bool(editing_pipe.config.is_distilled)

        # Passing dtype is essential: DiffusionPipeline.from_pipe() can otherwise
        # default to float32 and convert the SHARED model components in-place.
        inpaint_pipe = Flux2KleinInpaintPipeline.from_pipe(
            editing_pipe,
            dtype=model_dtype,
            is_distilled=is_distilled,
        )

        for component in (
            "transformer",
            "vae",
            "text_encoder",
            "tokenizer",
            "scheduler",
        ):
            if getattr(inpaint_pipe, component) is not getattr(editing_pipe, component):
                raise RuntimeError(
                    f"Klein inpaint initialization did not share '{component}' "
                    "with the editing pipeline. Refusing to use unexpected copies."
                )
        if bool(inpaint_pipe.config.is_distilled) != is_distilled:
            raise RuntimeError(
                "Klein inpaint pipeline lost its distilled/base configuration"
            )
        return inpaint_pipe

    @cached_property
    def mice_pipeline(self) -> Any:
        """MICE uses custom attention processors and requires separate weights."""
        from .external.mice_adapter import load_mice_pipeline

        return load_mice_pipeline(
            model_id=self.FLUX2_KLEIN_MODEL_ID,
            device=self.device,
            dtype=torch_dtype(
                self.device, cuda_dtype=torch.bfloat16, mps_dtype=torch.float32
            ),
        )

    @cached_property
    def pctnet_pipeline(self):
        from .external.pctnet_harmonization import load_pctnet

        return load_pctnet(device="cpu", variant="CNN")

    @cached_property
    def vitmatte_model(self):
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        processor = VitMatteImageProcessor.from_pretrained(self.VITMATTE_MODEL_ID)
        model = VitMatteForImageMatting.from_pretrained(self.VITMATTE_MODEL_ID)
        return processor, model.to(self.device, torch_dtype(self.device)).eval()

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
    def qwen_locate_pipelines(self) -> QwenLocatePipelines:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch_dtype(self.device)
        return QwenLocatePipelines(
            model=AutoModelForImageTextToText.from_pretrained(
                self.QWEN_LOCATE_MODEL_ID,
                torch_dtype=dtype,
            )
            .to(self.device)
            .eval(),
            processor=AutoProcessor.from_pretrained(self.QWEN_LOCATE_MODEL_ID),
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
