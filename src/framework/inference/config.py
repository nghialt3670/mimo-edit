from __future__ import annotations

from dataclasses import dataclass, field

from .utils import (
    validate_non_negative_integer,
    validate_non_negative_number,
    validate_number_range,
    validate_positive_integer,
)


@dataclass(frozen=True)
class KleinInferenceConfig:
    """Inference settings for the fixed FLUX.2 Klein base 4B backend."""

    num_inference_steps: int = 10
    guidance_scale: float = 4.0
    inpaint_strength: float = 1.0
    inpaint_padding_mask_crop: int = 64
    edit_seed: int = 42

    def __post_init__(self) -> None:
        validate_positive_integer("klein.num_inference_steps", self.num_inference_steps)
        validate_non_negative_number("klein.guidance_scale", self.guidance_scale)
        validate_number_range("klein.inpaint_strength", self.inpaint_strength, 0, 1)
        validate_positive_integer(
            "klein.inpaint_padding_mask_crop", self.inpaint_padding_mask_crop
        )
        validate_non_negative_integer("klein.edit_seed", self.edit_seed)


@dataclass(frozen=True)
class Sam3InferenceConfig:
    """Inference settings for the fixed SAM 3 segmentation model."""

    mask_threshold: float = 0.0

    def __post_init__(self) -> None:
        validate_number_range("sam3.mask_threshold", self.mask_threshold, 0, 1)


@dataclass(frozen=True)
class ObjectClearInferenceConfig:
    """Inference settings for the fixed ObjectClear removal model."""

    num_inference_steps: int = 8
    guidance_scale: float = 2.0
    seed: int = 42
    working_min_side: int = 384

    def __post_init__(self) -> None:
        validate_positive_integer(
            "object_clear.num_inference_steps", self.num_inference_steps
        )
        validate_non_negative_number("object_clear.guidance_scale", self.guidance_scale)
        validate_non_negative_integer("object_clear.seed", self.seed)
        validate_positive_integer(
            "object_clear.working_min_side", self.working_min_side
        )


@dataclass(frozen=True)
class InferenceConfig:
    """Inference settings for the fixed Klein, SAM 3, and ObjectClear backbones."""

    klein: KleinInferenceConfig = field(default_factory=KleinInferenceConfig)
    sam3: Sam3InferenceConfig = field(default_factory=Sam3InferenceConfig)
    object_clear: ObjectClearInferenceConfig = field(
        default_factory=ObjectClearInferenceConfig
    )
    low_vram: bool = False

    def __post_init__(self) -> None:
        for name, value, config_type in (
            ("klein", self.klein, KleinInferenceConfig),
            ("sam3", self.sam3, Sam3InferenceConfig),
            ("object_clear", self.object_clear, ObjectClearInferenceConfig),
        ):
            if not isinstance(value, config_type):
                raise TypeError(f"{name} must be a {config_type.__name__}")
