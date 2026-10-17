from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

Role = Literal["target", "anchor", "source", "context"]


class ParsedEntity(BaseModel):
    span: str = Field(description="Exact verbatim span copied from the instruction.")
    occurrence: int = Field(
        default=0, ge=0, description="0-based occurrence among exact span matches."
    )
    role: Role
    grounding_query: str = Field(
        description="Short phrase describing the visible source-image entity."
    )
    present_in_source: bool = True


class ParsedInstruction(BaseModel):
    operation: str
    entities: list[ParsedEntity] = Field(default_factory=list)
    mask_guidance_needed: bool = Field(
        default=False,
        description="Whether the text alone leaves the visible reference ambiguous and needs the benchmark mask.",
    )


class MaskClauseAssignment(BaseModel):
    mask_index: int = Field(description="1-based number of the highlighted region.")
    clause_index: int = Field(description="1-based number of the matching clause.")
    span: str = Field(
        description="Exact verbatim span copied from that clause naming the source object."
    )
    role: Role
    grounding_query: str = Field(
        description="Short phrase describing the visible source-image entity."
    )


class MaskClauseAssignments(BaseModel):
    assignments: list[MaskClauseAssignment] = Field(default_factory=list)


class Region(BaseModel):
    bbox_xyxy: tuple[float, float, float, float]
    image_size: tuple[int, int]
    confidence: float = Field(ge=0.0)
    label: str
    source: str
    crop_path: str | None = None
    mask_path: str | None = None


class Reference(BaseModel):
    token: str
    entity_id: str
    span: str
    start: int
    end: int
    role: Role
    grounding_query: str
    region: Region


class AnnotationRecord(BaseModel):
    sample_id: str
    benchmark: str
    split: str
    original_instruction: str
    mixed_instruction: str
    references: list[Reference]
    parser_instruction: str | None = None
    source_ref: str | None = None
    target_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    linguistic_stats: dict[str, float | int] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class Sample:
    sample_id: str
    benchmark: str
    split: str
    image: Image.Image
    instruction: str
    target: Image.Image | None = None
    mask: Image.Image | None = None
    source_ref: str | None = None
    target_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_instruction: str | None = None
    visible_reference_count: int = 0
    reference_masks: dict[str, Image.Image] = field(default_factory=dict)
