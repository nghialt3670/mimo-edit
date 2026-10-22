from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .utils import (
    validate_non_negative_number,
    validate_number_range,
    validate_positive_integer,
)

KleinRemoveLora = Literal["fal", "slopshop"]
KLEIN_REMOVE_LORAS = ("fal", "slopshop")


@dataclass(frozen=True)
class KleinInferenceConfig:
    """Inference settings for the fixed FLUX.2 Klein base 4B backend."""

    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    mask_context_padding: int = 64
    # ``fal``: magenta-highlight object-remove LoRA. ``slopshop``: erase-v1
    # with the original photo plus an RGB mask as the second reference image.
    remove_lora: KleinRemoveLora = "fal"

    def __post_init__(self) -> None:
        validate_positive_integer("klein.num_inference_steps", self.num_inference_steps)
        validate_non_negative_number("klein.guidance_scale", self.guidance_scale)
        validate_positive_integer(
            "klein.mask_context_padding", self.mask_context_padding
        )
        if self.remove_lora not in KLEIN_REMOVE_LORAS:
            raise ValueError(
                "klein.remove_lora must be one of: " + ", ".join(KLEIN_REMOVE_LORAS)
            )


@dataclass(frozen=True)
class Sam3InferenceConfig:
    """Inference settings for the fixed SAM 3 segmentation model."""

    mask_threshold: float = 0.0

    def __post_init__(self) -> None:
        validate_number_range("sam3.mask_threshold", self.mask_threshold, 0, 1)


@dataclass(frozen=True)
class GroundingDinoInferenceConfig:
    """Inference settings for the fixed Grounding DINO detector."""

    box_threshold: float = 0.35
    text_threshold: float = 0.25

    def __post_init__(self) -> None:
        validate_number_range("grounding_dino.box_threshold", self.box_threshold, 0, 1)
        validate_number_range(
            "grounding_dino.text_threshold", self.text_threshold, 0, 1
        )


@dataclass(frozen=True)
class ObjectClearInferenceConfig:
    """Inference settings for the fixed ObjectClear removal model."""

    num_inference_steps: int = 8
    guidance_scale: float = 2.0
    working_min_side: int = 384
    # The input mask is dilated by this fraction of its bounding-box longer side.
    mask_padding_ratio: float = 0.05

    def __post_init__(self) -> None:
        validate_non_negative_number(
            "object_clear.mask_padding_ratio", self.mask_padding_ratio
        )
        validate_positive_integer(
            "object_clear.num_inference_steps", self.num_inference_steps
        )
        validate_non_negative_number("object_clear.guidance_scale", self.guidance_scale)
        validate_positive_integer(
            "object_clear.working_min_side", self.working_min_side
        )


@dataclass(frozen=True)
class ViTMatteInferenceConfig:
    """Inference settings for the fixed ViTMatte alpha-matting model."""

    # Longest side of the matting input; larger crops are downscaled.
    max_side: int = 1536

    def __post_init__(self) -> None:
        validate_positive_integer("vitmatte.max_side", self.max_side)


@dataclass(frozen=True)
class InferenceConfig:
    """Inference settings for the fixed image-editing and segmentation backbones."""

    klein: KleinInferenceConfig = field(default_factory=KleinInferenceConfig)
    sam3: Sam3InferenceConfig = field(default_factory=Sam3InferenceConfig)
    grounding_dino: GroundingDinoInferenceConfig = field(
        default_factory=GroundingDinoInferenceConfig
    )
    object_clear: ObjectClearInferenceConfig = field(
        default_factory=ObjectClearInferenceConfig
    )
    vitmatte: ViTMatteInferenceConfig = field(default_factory=ViTMatteInferenceConfig)
    low_vram: bool = False

    def __post_init__(self) -> None:
        for name, value, config_type in (
            ("klein", self.klein, KleinInferenceConfig),
            ("sam3", self.sam3, Sam3InferenceConfig),
            ("grounding_dino", self.grounding_dino, GroundingDinoInferenceConfig),
            ("object_clear", self.object_clear, ObjectClearInferenceConfig),
            ("vitmatte", self.vitmatte, ViTMatteInferenceConfig),
        ):
            if not isinstance(value, config_type):
                raise TypeError(f"{name} must be a {config_type.__name__}")
