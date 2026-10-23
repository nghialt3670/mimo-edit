"""Re-mask RefEdit mixed annotations when the dataset mask covers a host object."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from visualization.prompts import build_visual_prompts

from .core import (
    AnnotationError,
    linguistic_stats,
    locate_spans,
    replace_spans,
    validate_record,
)
from .grounder import mask_bbox, resize_mask
from .io import append_jsonl, save_reference_assets
from .schema import (
    AnnotationRecord,
    ParsedEntity,
    Reference,
    RefinementDecision,
    Region,
    Sample,
)


_IDENTIFIER_PREFIX = re.compile(
    r"^\s+(?:from|of|on|in|at|next to|beside|behind|in front of|"
    r"worn by|held by|that|who|which)\b",
    re.IGNORECASE,
)
_REQUESTED_CHANGE = re.compile(
    r"\s+(?:to|into|with|so that|so it)\b",
    re.IGNORECASE,
)


def include_identifying_tail(text: str, start: int, end: int) -> tuple[int, int]:
    """Fold host/identifier phrases into the span; keep requested-change wording out.

    Visual prompts already point at the object, so mixed instructions should not
    keep 'from the man ...' after <v1>.
    """
    rest = text[end:]
    trailing = 0
    while trailing < len(rest) and rest[-(trailing + 1)] in ".!?":
        trailing += 1
    body = rest[: len(rest) - trailing] if trailing else rest
    if not _IDENTIFIER_PREFIX.match(body):
        return start, end
    change = _REQUESTED_CHANGE.search(body)
    identifier = body[: change.start()] if change else body
    if not identifier.strip():
        return start, end
    return start, end + len(identifier)


class RefinementParser(Protocol):
    def propose_refinement(
        self,
        *,
        original_instruction: str,
        mixed_instruction: str,
        span: str,
        role: str,
        grounding_query: str,
        image: Image.Image,
        mask: Image.Image,
    ) -> RefinementDecision: ...


class SegmentText(Protocol):
    def __call__(self, image: Image.Image, text: str) -> list[Image.Image]: ...


def refined_output_path(annotations: Path) -> Path:
    return annotations.with_name(f"refined_{annotations.name}")


def refined_audit_path(annotations: Path) -> Path:
    name = annotations.name.replace("_mixed.jsonl", "_audit.jsonl")
    if name == annotations.name:
        name = f"{annotations.stem}_audit.jsonl"
    return annotations.with_name(f"refined_{name}")


def load_reference_mask(
    record: dict[str, Any], annotations_dir: Path
) -> Image.Image | None:
    references = record.get("references") or []
    if not references:
        return None
    region = references[0].get("region") or {}
    mask_path = region.get("mask_path")
    if not mask_path:
        return None
    path = Path(mask_path)
    path = path if path.is_absolute() else annotations_dir / path
    if not path.is_file():
        return None
    with Image.open(path) as mask:
        loaded = mask.convert("L")
        image_size = region.get("image_size")
        if (
            isinstance(image_size, (list, tuple))
            and len(image_size) == 2
            and loaded.size != (int(image_size[0]), int(image_size[1]))
        ):
            loaded = loaded.resize(
                (int(image_size[0]), int(image_size[1])), Image.Resampling.NEAREST
            )
        return loaded


def select_sam3_mask(
    candidates: list[Image.Image], host_mask: Image.Image | None
) -> Image.Image:
    """Prefer a SAM 3 mask that lies inside the original host mask."""
    if not candidates:
        raise ValueError("SAM 3 returned no masks")
    if host_mask is None:
        return candidates[0]
    host = np.asarray(resize_mask(host_mask, candidates[0].size)) > 0
    scored: list[tuple[float, Image.Image]] = []
    for mask in candidates:
        array = np.asarray(resize_mask(mask, candidates[0].size)) > 0
        overlap = float((array & host).sum()) / max(int(array.sum()), 1)
        scored.append((overlap, mask))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] > 0.3:
        return scored[0][1]
    return candidates[0]


def apply_refinement(
    record: dict[str, Any],
    image: Image.Image,
    mask: Image.Image,
    decision: RefinementDecision,
) -> AnnotationRecord:
    """Rebuild the mixed instruction and reference around a new SAM 3 mask."""
    original = str(record["original_instruction"])
    entity = ParsedEntity(
        span=decision.span,
        occurrence=decision.occurrence,
        role=decision.role,
        grounding_query=decision.grounding_query or decision.sam3_text or decision.span,
    )
    located = locate_spans(original, [entity])
    start, end, located_entity = located[0]
    start, end = include_identifying_tail(original, start, end)
    located_entity = located_entity.model_copy(update={"span": original[start:end]})
    located = [(start, end, located_entity)]
    mixed = replace_spans(original, located)
    bbox = mask_bbox(mask, image.size)
    if bbox is None:
        raise AnnotationError("refined_mask_empty")
    sample_id = str(record["sample_id"])
    rebuilt = AnnotationRecord(
        sample_id=sample_id,
        benchmark=str(record.get("benchmark", "refedit_bench")),
        split=str(record.get("split", "")),
        original_instruction=original,
        parser_instruction=record.get("parser_instruction"),
        mixed_instruction=mixed,
        references=[
            Reference(
                token="<v1>",
                entity_id="e1",
                span=located_entity.span,
                start=start,
                end=end,
                role=located_entity.role,
                grounding_query=located_entity.grounding_query,
                region=Region(
                    bbox_xyxy=bbox,
                    image_size=image.size,
                    confidence=1.0,
                    label=located_entity.grounding_query,
                    source="sam3:text",
                ),
            )
        ],
        source_ref=record.get("source_ref"),
        target_ref=record.get("target_ref"),
        metadata={
            **(record.get("metadata") or {}),
            "refined": True,
            "refinement": {
                "reason": decision.reason,
                "sam3_text": decision.sam3_text,
                "previous_span": (record.get("references") or [{}])[0].get("span"),
                "previous_mixed_instruction": record.get("mixed_instruction"),
            },
        },
        linguistic_stats=linguistic_stats(original, mixed),
    )
    rebuilt.validation = validate_record(rebuilt, mixed)
    if not rebuilt.validation["pass"]:
        raise AnnotationError("refinement_validation_failed")
    return rebuilt


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if sample_id is not None:
                ids.add(str(sample_id))
    return ids


def refine_annotations(
    annotations_path: Path,
    *,
    parser: RefinementParser,
    load_image: Callable[[dict[str, Any]], Image.Image],
    segment_text: SegmentText,
    output_path: Path | None = None,
    audit_path: Path | None = None,
    sample_ids: set[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Review every mixed annotation and re-mask samples the LLM marks as optimizable."""
    annotations_path = annotations_path.expanduser().resolve()
    output_path = (output_path or refined_output_path(annotations_path)).resolve()
    audit_path = (audit_path or refined_audit_path(annotations_path)).resolve()
    annotations_dir = annotations_path.parent

    records: list[dict[str, Any]] = []
    with annotations_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {annotations_path} at line {line_number}"
                ) from exc

    already_written = _existing_ids(output_path)
    counts = {
        "records": len(records),
        "copied": 0,
        "optimized": 0,
        "skipped": 0,
        "failed": 0,
        "resumed": 0,
    }
    attempted = 0

    for record in tqdm(records, desc="refine"):
        sample_id = str(record.get("sample_id", ""))
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        if sample_id in already_written:
            counts["resumed"] += 1
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1

        references = record.get("references") or []
        if len(references) != 1:
            _write_copy(record, output_path, dry_run)
            counts["copied"] += 1
            counts["skipped"] += 1
            continue

        try:
            image = load_image(record)
            mask = load_reference_mask(record, annotations_dir)
            if mask is None:
                raise AnnotationError("missing_reference_mask")
            current = references[0]
            decision = parser.propose_refinement(
                original_instruction=str(record.get("original_instruction", "")),
                mixed_instruction=str(record.get("mixed_instruction", "")),
                span=str(current.get("span", "")),
                role=str(current.get("role", "target")),
                grounding_query=str(current.get("grounding_query", "")),
                image=image,
                mask=mask,
            )
            if not decision.optimize or not decision.span.strip():
                _write_copy(record, output_path, dry_run)
                counts["copied"] += 1
                continue
            if not decision.sam3_text.strip():
                raise AnnotationError("refinement_missing_sam3_text")
            if dry_run:
                counts["optimized"] += 1
                continue

            candidates = segment_text(image, decision.sam3_text)
            new_mask = select_sam3_mask(candidates, mask)
            rebuilt = apply_refinement(record, image, new_mask, decision)
            sample = Sample(
                sample_id=rebuilt.sample_id,
                benchmark=rebuilt.benchmark,
                split=rebuilt.split,
                image=image,
                instruction=rebuilt.original_instruction,
                mask=new_mask,
                source_ref=rebuilt.source_ref,
                target_ref=rebuilt.target_ref,
                metadata=rebuilt.metadata,
                reference_masks={"e1": new_mask},
            )
            save_reference_assets(sample, rebuilt, output_path)
            payload = rebuilt.model_dump(mode="json")
            for reference in payload["references"]:
                reference["visual_prompts"] = build_visual_prompts(
                    reference, output_path.parent, rebuilt.sample_id
                )
            append_jsonl(output_path, payload)
            already_written.add(sample_id)
            counts["optimized"] += 1
        except Exception as exc:
            counts["failed"] += 1
            if not dry_run:
                append_jsonl(
                    audit_path,
                    {
                        "status": "failed",
                        "sample_id": sample_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            _write_copy(record, output_path, dry_run)
            counts["copied"] += 1

    return counts


def _write_copy(record: dict[str, Any], output_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    append_jsonl(output_path, record)
