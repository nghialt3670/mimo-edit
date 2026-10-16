from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from PIL import Image

from .grounder import Grounder, mask_bbox, mask_components
from .parser import Parser
from .schema import (
    AnnotationRecord,
    ParsedEntity,
    ParsedInstruction,
    Reference,
    Region,
    Sample,
)

TOKEN_RE = re.compile(r"<v\d+>")
WORD_RE = re.compile(r"\b[\w'-]+\b")
OBJECT_COUNT_RE = re.compile(
    r"\b(?P<count>\d+|two|three|four|five|six|seven|eight|nine|ten|both)\b",
    re.IGNORECASE,
)
LEADING_SPATIAL_PREPOSITION_RE = re.compile(
    r"^(?:in front of|to the (?:left|right) of|in|on|at|to|from|between|"
    r"below|above|beside|near|under|over|behind)\b\s+",
    re.IGNORECASE,
)


class AnnotationError(RuntimeError):
    pass


class NoReferenceError(AnnotationError):
    def __init__(
        self,
        message: str,
        *,
        parse_attempts: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(message)
        self.parse_attempts = parse_attempts or []


def _parse_attempt_details(mode: str, parsed: ParsedInstruction) -> dict[str, object]:
    entities = parsed.entities
    return {
        "mode": mode,
        "entity_count": len(entities),
        "entities": [
            {
                "span": entity.span,
                "role": entity.role,
                "present_in_source": entity.present_in_source,
            }
            for entity in entities
        ],
    }


def _recover_span(text: str, span: str) -> tuple[int, int] | None:
    source_words = list(WORD_RE.finditer(text))
    query_words = [word.casefold() for word in WORD_RE.findall(span)]
    if not query_words:
        return None

    candidates: list[tuple[float, int, int]] = []
    minimum_length = max(1, len(query_words) - 2)
    maximum_length = len(query_words) + 2
    for start_index in range(len(source_words)):
        for length in range(
            minimum_length, min(maximum_length, len(source_words) - start_index) + 1
        ):
            end_index = start_index + length
            candidate_words = [
                word.group().casefold() for word in source_words[start_index:end_index]
            ]
            similarity = SequenceMatcher(
                None, query_words, candidate_words, autojunk=False
            ).ratio()
            candidates.append(
                (
                    similarity,
                    source_words[start_index].start(),
                    source_words[end_index - 1].end(),
                )
            )

    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] < 0.9:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.05:
        return None
    return candidates[0][1], candidates[0][2]


def locate_spans(
    text: str, entities: list[ParsedEntity]
) -> list[tuple[int, int, ParsedEntity]]:
    located: list[tuple[int, int, ParsedEntity]] = []
    for entity in entities:
        if not entity.span:
            stripped = text.rstrip()
            end = len(stripped)
            while end and stripped[end - 1] in ".,;:!?":
                end -= 1
            located.append((end, end, entity))
            continue
        matches = list(re.finditer(re.escape(entity.span), text))
        if entity.occurrence >= len(matches):
            recovered = (
                _recover_span(text, entity.span) if entity.occurrence == 0 else None
            )
            if recovered is not None:
                start, end = recovered
                corrected_entity = entity.model_copy(update={"span": text[start:end]})
                located.append((start, end, corrected_entity))
                continue
            raise AnnotationError(
                f"span_not_found: span={entity.span!r} occurrence={entity.occurrence} matches={len(matches)}"
            )
        match = matches[entity.occurrence]
        located.append((match.start(), match.end(), entity))
    located.sort(key=lambda x: (x[0], x[1]))
    for left, right in zip(located, located[1:]):
        if left[1] > right[0]:
            raise AnnotationError(
                f"overlapping_spans:{left[2].span!r}|{right[2].span!r}"
            )
    return located


def _locate_spans_allow_same(
    text: str, entities: list[ParsedEntity]
) -> list[tuple[int, int, ParsedEntity]]:
    located = [locate_spans(text, [entity])[0] for entity in entities]
    located.sort(key=lambda item: (item[0], item[1]))
    for left, right in zip(located, located[1:]):
        if left[1] > right[0] and (left[0], left[1]) != (right[0], right[1]):
            raise AnnotationError(
                f"overlapping_spans:{left[2].span!r}|{right[2].span!r}"
            )
    return located


def _expected_object_count(instruction: str, fallback: int) -> int:
    match = OBJECT_COUNT_RE.search(instruction)
    if match is None:
        return max(1, fallback)
    value = match.group("count").casefold()
    if value == "both":
        return 2
    if value.isdigit():
        return max(1, int(value))
    return {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }[value]


def _caption_object_count(caption: str, fallback: int) -> int:
    count = len([part for part in caption.split("|") if part.strip()])
    return count or max(1, fallback)


def _normalize_add_mask_spans(
    text: str,
    entries: list[tuple[ParsedEntity, Image.Image, Region]],
) -> list[tuple[ParsedEntity, Image.Image, Region]]:
    entries = list(entries)
    while True:
        located = [locate_spans(text, [entity])[0] for entity, _, _ in entries]
        changed = False
        for first_index, first in enumerate(located):
            first_span = first[:2]
            for second_index in range(first_index + 1, len(located)):
                second_span = located[second_index][:2]
                if first_span[1] <= second_span[0] or second_span[1] <= first_span[0]:
                    continue
                if first_span == second_span:
                    continue

                longer_index: int | None = None
                shorter_index: int | None = None
                if first_span[0] <= second_span[0] and second_span[1] <= first_span[1]:
                    longer_index, shorter_index = first_index, second_index
                elif (
                    second_span[0] <= first_span[0] and first_span[1] <= second_span[1]
                ):
                    longer_index, shorter_index = second_index, first_index

                if longer_index is not None and shorter_index is not None:
                    longer = located[longer_index][:2]
                    shorter = located[shorter_index][:2]
                    extra_prefix = text[longer[0] : shorter[0]]
                    extra_suffix = text[shorter[1] : longer[1]]
                    trims_spatial_lead = bool(
                        LEADING_SPATIAL_PREPOSITION_RE.fullmatch(extra_prefix)
                        or re.fullmatch(
                            r"(?:in front of|to the (?:left|right) of|in|on|at|to|"
                            r"from|between|below|above|beside|near|under|over|behind)\s+",
                            extra_prefix,
                            re.IGNORECASE,
                        )
                    )
                    trims_conjoined_tail = bool(
                        re.match(r"^\s*(?:and|or|,)\b", extra_suffix, re.IGNORECASE)
                    )
                    if trims_spatial_lead or trims_conjoined_tail:
                        entity, mask, region = entries[longer_index]
                        shorter_entity = entries[shorter_index][0]
                        entries[longer_index] = (
                            entity.model_copy(
                                update={
                                    "span": shorter_entity.span,
                                    "occurrence": shorter_entity.occurrence,
                                }
                            ),
                            mask,
                            region,
                        )
                        changed = True
                        break

                raise AnnotationError(
                    "multi_object_add_overlapping_mask_spans:"
                    f"{entries[first_index][0].span!r}|"
                    f"{entries[second_index][0].span!r}"
                )
            if changed:
                break
        if not changed:
            return entries


def _single_span_is_unambiguous(text: str, entities: list[ParsedEntity]) -> bool:
    if len(entities) != 1:
        return False
    entity = entities[0]
    matches = list(re.finditer(re.escape(entity.span), text))
    return len(matches) == 1 and entity.occurrence == 0


def _trim_conjoined_overlap(
    text: str,
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, str] | None:
    first_start, first_end = first
    second_start, second_end = second
    if first_start <= second_start and second_end == first_end:
        prefix = text[first_start:second_start]
        prefix = re.sub(r"\s+(?:and|or|,)\s*$", "", prefix, flags=re.IGNORECASE).strip()
        if prefix:
            return 0, prefix
    if second_start <= first_start and first_end == second_end:
        prefix = text[second_start:first_start]
        prefix = re.sub(r"\s+(?:and|or|,)\s*$", "", prefix, flags=re.IGNORECASE).strip()
        if prefix:
            return 1, prefix
    if first_start == second_start and first_end < second_end:
        suffix = text[first_end:second_end]
        suffix = re.sub(r"^\s*(?:and|or|,)\s*", "", suffix, flags=re.IGNORECASE).strip()
        if suffix:
            return 1, suffix
    if second_start == first_start and second_end < first_end:
        suffix = text[second_end:first_end]
        suffix = re.sub(r"^\s*(?:and|or|,)\s*", "", suffix, flags=re.IGNORECASE).strip()
        if suffix:
            return 0, suffix
    return None


def _disambiguate_masked_spans(
    text: str,
    entries: list[tuple[ParsedEntity, Image.Image, Region]],
) -> list[tuple[ParsedEntity, Image.Image, Region]]:
    entries = list(entries)
    while True:
        spans = [locate_spans(text, [entity])[0][:2] for entity, _, _ in entries]
        changed = False
        for first_index, first in enumerate(spans):
            for second_index in range(first_index + 1, len(spans)):
                second = spans[second_index]
                if first[1] <= second[0] or second[1] <= first[0]:
                    continue
                if first == second:
                    continue
                trimmed = _trim_conjoined_overlap(text, first, second)
                if trimmed is None:
                    raise AnnotationError(
                        f"mask_guided_overlapping_spans:"
                        f"{entries[first_index][0].span!r}|{entries[second_index][0].span!r}"
                    )
                entry_index, span = trimmed
                entity, mask, region = entries[entry_index]
                entries[entry_index] = (
                    entity.model_copy(update={"span": span, "occurrence": 0}),
                    mask,
                    region,
                )
                changed = True
                break
            if changed:
                break
        if not changed:
            return entries


def _assign_repeated_masked_occurrences(
    text: str,
    entries: list[tuple[ParsedEntity, Image.Image, Region]],
) -> list[tuple[ParsedEntity, Image.Image, Region]]:
    indices_by_span: dict[str, list[int]] = {}
    for index, (entity, _, _) in enumerate(entries):
        indices_by_span.setdefault(entity.span, []).append(index)

    for span, indices in indices_by_span.items():
        if len(indices) < 2 or any(
            entries[index][0].occurrence != 0 for index in indices
        ):
            continue
        matches = list(re.finditer(re.escape(span), text))
        if len(matches) < len(indices):
            continue
        for occurrence, index in enumerate(indices):
            entity, mask, region = entries[index]
            entries[index] = (
                entity.model_copy(update={"occurrence": occurrence}),
                mask,
                region,
            )
    return entries


def replace_spans(
    text: str,
    located: list[tuple[int, int, ParsedEntity]],
    *,
    spatial_destination: bool = False,
) -> str:
    pieces: list[str] = []
    cursor = 0
    index = 0
    while index < len(located):
        start, end, _ = located[index]
        replacements: list[str] = []
        while index < len(located) and located[index][:2] == (start, end):
            _, _, duplicate_entity = located[index]
            replacement = f"<v{index + 1}>"
            needs_destination_preposition = not replacements
            if not duplicate_entity.span:
                replacement = (
                    f" at {replacement}"
                    if needs_destination_preposition
                    else replacement
                )
            elif spatial_destination and LEADING_SPATIAL_PREPOSITION_RE.match(
                duplicate_entity.span
            ):
                replacement = (
                    f"at {replacement}"
                    if needs_destination_preposition
                    else replacement
                )
            replacements.append(replacement)
            index += 1
        pieces.extend((text[cursor:start], " and ".join(replacements)))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def linguistic_stats(original: str, mixed: str) -> dict[str, float | int]:
    original_words = len(WORD_RE.findall(original))
    mixed_words = len([w for w in WORD_RE.findall(TOKEN_RE.sub("", mixed)) if w])
    removed_words = original_words - mixed_words
    reduction = 0.0 if original_words == 0 else 1.0 - mixed_words / original_words
    return {
        "original_words": original_words,
        "mixed_words": mixed_words,
        "reference_words_removed": removed_words,
        "reduction": round(reduction, 6),
    }


def validate_record(record: AnnotationRecord, expected_mixed: str) -> dict[str, object]:
    tokens = TOKEN_RE.findall(record.mixed_instruction)
    refs = [r.token for r in record.references]
    boxes_ok = all(
        0 <= r.region.bbox_xyxy[0] < r.region.bbox_xyxy[2] <= r.region.image_size[0]
        and 0 <= r.region.bbox_xyxy[1] < r.region.bbox_xyxy[3] <= r.region.image_size[1]
        for r in record.references
    )
    ok = (
        record.mixed_instruction == expected_mixed
        and tokens == refs
        and len(refs) == len(set(refs))
        and len(refs) > 0
        and boxes_ok
    )
    return {
        "pass": ok,
        "tokens": tokens,
        "reference_tokens": refs,
        "boxes_valid": boxes_ok,
        "source_target_preserved": True,
    }


@dataclass(slots=True)
class Annotator:
    parser: Parser
    grounder: Grounder
    min_confidence: float = 0.0
    refedit_mask_from_start: bool = False

    def annotate(self, sample: Sample) -> AnnotationRecord:
        parser_instruction = sample.parser_instruction or sample.instruction
        mira_bench_reference_masks = (
            dict(sample.reference_masks) if sample.benchmark == "MIRA-Bench" else {}
        )
        sample.reference_masks.clear()
        mira_bench_reference_only = sample.benchmark == "MIRA-Bench"
        masked_reference_only = sample.benchmark == "refedit_bench"
        multi_object_add = (
            sample.benchmark == "CompBench"
            and sample.metadata.get("task") == "multi_object_add"
        )
        multi_object_remove = (
            sample.benchmark == "CompBench"
            and sample.metadata.get("task") == "multi_object_remove"
        )
        mask_guided_spatial_reference = (
            sample.benchmark == "CompBench"
            and sample.metadata.get("task") in {"add", "multi_object_add"}
            and sample.mask is not None
            and sample.mask.getbbox() is not None
        )
        mask_guidance_used = False
        mask_guidance_reasons: list[str] = []
        parse_attempts: list[dict[str, object]] = []
        component_regions: dict[tuple[int, int], tuple[Region, Image.Image]] = {}
        component_region_queues: dict[
            tuple[int, int], list[tuple[Region, Image.Image]]
        ] = {}
        if masked_reference_only:
            if sample.mask is None or sample.mask.getbbox() is None:
                raise AnnotationError("refedit_bench_mask_missing_or_empty")
            if self.refedit_mask_from_start:
                parsed = self.parser.parse(
                    parser_instruction,
                    image=sample.image,
                    mask=sample.mask,
                    masked_reference_only=True,
                )
                parse_attempts.append(_parse_attempt_details("masked", parsed))
                entities = [e for e in parsed.entities if e.present_in_source]
                mask_guidance_used = True
                mask_guidance_reasons.append("mask_from_start")
            else:
                parsed = self.parser.parse(parser_instruction)
                text_parsed = parsed
                parse_attempts.append(_parse_attempt_details("text", parsed))
                entities = [e for e in parsed.entities if e.present_in_source]
                if parsed.mask_guidance_needed:
                    mask_guidance_reasons.append("parser_flag")
                if len(entities) != 1:
                    mask_guidance_reasons.append("reference_count")
                if not _single_span_is_unambiguous(parser_instruction, entities):
                    mask_guidance_reasons.append("span_alignment")
                if mask_guidance_reasons:
                    parsed = self.parser.parse(
                        parser_instruction,
                        image=sample.image,
                        mask=sample.mask,
                        masked_reference_only=True,
                    )
                    parse_attempts.append(_parse_attempt_details("masked", parsed))
                    entities = [e for e in parsed.entities if e.present_in_source]
                    mask_guidance_used = True
                    if not entities:
                        text_entities = [
                            entity
                            for entity in text_parsed.entities
                            if entity.present_in_source
                        ]
                        if _single_span_is_unambiguous(
                            parser_instruction, text_entities
                        ):
                            entities = text_entities
                            mask_guidance_reasons.append(
                                "masked_retry_empty_text_fallback"
                            )
        elif multi_object_add:
            if sample.mask is None or sample.mask.getbbox() is None:
                raise AnnotationError("multi_object_add_mask_missing_or_empty")
            caption = str(sample.metadata.get("caption", ""))
            initial_components = mask_components(sample.mask)
            fallback_count = max(
                _caption_object_count(caption, 2),
                len(initial_components),
            )
            expected_count = _expected_object_count(parser_instruction, fallback_count)
            components = mask_components(sample.mask, expected_count=expected_count)
            if len(components) < expected_count:
                raise AnnotationError(
                    "multi_object_add_mask_split_failed:"
                    f"expected={expected_count},got={len(components)}"
                )

            mask_entries: list[tuple[ParsedEntity, Image.Image, Region]] = []
            for _, component_mask in components[:expected_count]:
                parsed = self.parser.parse(
                    parser_instruction,
                    image=sample.image,
                    mask=component_mask,
                    mask_guided_spatial_reference=True,
                )
                component_entities = [e for e in parsed.entities if e.present_in_source]
                if len(component_entities) != 1:
                    if component_entities:
                        raise AnnotationError(
                            "multi_object_add_expected_one_mask_reference:"
                            f"got={len(component_entities)}"
                        )
                    component_entities = [
                        ParsedEntity(
                            span="",
                            role="target",
                            grounding_query="masked addition destination",
                        )
                    ]
                entity = component_entities[0]
                bbox = mask_bbox(component_mask, sample.image.size)
                if bbox is None:
                    raise AnnotationError("multi_object_add_empty_mask_component")
                mask_entries.append(
                    (
                        entity,
                        component_mask,
                        Region(
                            bbox_xyxy=bbox,
                            image_size=sample.image.size,
                            confidence=1.0,
                            label=entity.grounding_query,
                            source="benchmark_mask:edit_region_component",
                        ),
                    )
                )

            mask_entries = _normalize_add_mask_spans(
                parser_instruction,
                mask_entries,
            )
            entities = []
            for entity, component_mask, region in mask_entries:
                start, end, _ = locate_spans(parser_instruction, [entity])[0]
                component_region_queues.setdefault((start, end), []).append(
                    (region, component_mask)
                )
                entities.append(entity)
            mask_guidance_used = True
            mask_guidance_reasons.append("compbench_multi_object_add_mask_regions")
        elif mira_bench_reference_only:
            if any(
                f"e{index}" not in mira_bench_reference_masks for index in range(1, 6)
            ):
                raise AnnotationError("mira_bench_reference_masks_missing")

            mask_entries: list[tuple[ParsedEntity, Image.Image, Region]] = []
            for index in range(1, 6):
                mask = mira_bench_reference_masks[f"e{index}"]
                entity: ParsedEntity | None = None
                parse_problem = "no_source_reference"
                source_entity_count = 0
                for retry in (False, True):
                    parsed = self.parser.parse(
                        parser_instruction,
                        image=sample.image,
                        mask=mask,
                        masked_reference_only=True,
                        mira_bench_masked_reference=retry,
                    )
                    mode = f"masked_region_{index}"
                    if retry:
                        mode += "_retry"
                    parse_attempts.append(_parse_attempt_details(mode, parsed))
                    component_entities = [
                        candidate
                        for candidate in parsed.entities
                        if candidate.present_in_source
                    ]
                    source_entity_count = len(component_entities)
                    if len(component_entities) != 1:
                        parse_problem = f"expected_one_source_reference_got_{len(component_entities)}"
                        continue
                    try:
                        locate_spans(parser_instruction, component_entities)
                    except AnnotationError as exc:
                        parse_problem = str(exc)
                        continue
                    entity = component_entities[0]
                    break

                if entity is None:
                    if source_entity_count == 0:
                        raise NoReferenceError(
                            f"mira_bench_mask_region_no_source_reference:e{index}",
                            parse_attempts=parse_attempts,
                        )
                    raise AnnotationError(
                        f"mira_bench_mask_region_parse_failed:e{index}:{parse_problem}"
                    )

                bbox = mask_bbox(mask, sample.image.size)
                if bbox is None:
                    raise AnnotationError(f"mira_bench_reference_mask_empty:e{index}")
                mask_entries.append(
                    (
                        entity,
                        mask,
                        Region(
                            bbox_xyxy=bbox,
                            image_size=sample.image.size,
                            confidence=1.0,
                            label=entity.grounding_query,
                            source="benchmark_mask:object_instance",
                        ),
                    )
                )

            entities = []
            mask_entries = _assign_repeated_masked_occurrences(
                parser_instruction,
                mask_entries,
            )
            for entity, mask, region in _disambiguate_masked_spans(
                parser_instruction, mask_entries
            ):
                start, end, _ = locate_spans(parser_instruction, [entity])[0]
                component_region_queues.setdefault((start, end), []).append(
                    (region, mask)
                )
                entities.append(entity)
            mask_guidance_used = True
            mask_guidance_reasons.append("mira_bench_reference_masks")
        elif multi_object_remove:
            if sample.mask is None or sample.mask.getbbox() is None:
                raise AnnotationError("multi_object_remove_mask_missing_or_empty")
            initial_components = mask_components(sample.mask)
            caption = str(sample.metadata.get("caption", ""))
            fallback_count = max(
                _caption_object_count(caption, 2),
                len(initial_components),
            )
            expected_count = _expected_object_count(parser_instruction, fallback_count)
            components = mask_components(sample.mask, expected_count=expected_count)
            if len(components) < expected_count:
                raise AnnotationError(
                    "multi_object_remove_mask_split_failed:"
                    f"expected={expected_count},got={len(components)}"
                )

            mask_entries: list[tuple[ParsedEntity, Image.Image, Region]] = []
            for _, component_mask in components[:expected_count]:
                parsed = self.parser.parse(
                    parser_instruction,
                    image=sample.image,
                    mask=component_mask,
                    masked_reference_only=True,
                )
                component_entities = [e for e in parsed.entities if e.present_in_source]
                if len(component_entities) != 1:
                    raise AnnotationError(
                        "multi_object_remove_expected_one_mask_reference:"
                        f"got={len(component_entities)}"
                    )

                entity = component_entities[0]
                bbox = mask_bbox(component_mask, sample.image.size)
                if bbox is None:
                    raise AnnotationError("multi_object_remove_empty_mask_component")
                mask_entries.append(
                    (
                        entity,
                        component_mask,
                        Region(
                            bbox_xyxy=bbox,
                            image_size=sample.image.size,
                            confidence=1.0,
                            label=entity.grounding_query,
                            source="benchmark_mask:edit_region_component",
                        ),
                    )
                )

            entities = []
            for entity, component_mask, region in _disambiguate_masked_spans(
                parser_instruction,
                mask_entries,
            ):
                start, end, _ = locate_spans(parser_instruction, [entity])[0]
                key = (start, end)
                component_region_queues.setdefault(key, []).append(
                    (region, component_mask)
                )
                entities.append(entity)
            mask_guidance_used = True
            mask_guidance_reasons.append("compbench_multi_object_remove_mask_regions")
        elif mask_guided_spatial_reference:
            if sample.mask is None:
                raise AnnotationError("compbench_add_mask_missing_or_empty")
            parsed = self.parser.parse(
                parser_instruction,
                image=sample.image,
                mask=sample.mask,
                mask_guided_spatial_reference=True,
            )
            entities = [e for e in parsed.entities if e.present_in_source]
            if len(entities) > 1:
                raise AnnotationError(
                    f"compbench_add_expected_one_mask_reference:got={len(entities)}"
                )
            mask_guidance_used = True
            mask_guidance_reasons.append("compbench_add_edit_mask")
            if sample.metadata.get("task") == "add" and not entities:
                entity = ParsedEntity(
                    span="",
                    role="target",
                    grounding_query="masked addition destination",
                )
                insertion = locate_spans(parser_instruction, [entity])[0][0]
                bbox = mask_bbox(sample.mask, sample.image.size)
                if bbox is None:
                    raise AnnotationError("compbench_add_mask_missing_or_empty")
                entities = [entity]
                component_regions[(insertion, insertion)] = (
                    Region(
                        bbox_xyxy=bbox,
                        image_size=sample.image.size,
                        confidence=1.0,
                        label=entity.grounding_query,
                        source="benchmark_mask:edit_region",
                    ),
                    sample.mask,
                )
                mask_guidance_reasons.append("compbench_add_no_location_phrase")
            elif entities:
                entity = entities[0]
                start, end, _ = locate_spans(parser_instruction, [entity])[0]
                bbox = mask_bbox(sample.mask, sample.image.size)
                if bbox is None:
                    raise AnnotationError("compbench_add_mask_missing_or_empty")
                component_regions[(start, end)] = (
                    Region(
                        bbox_xyxy=bbox,
                        image_size=sample.image.size,
                        confidence=1.0,
                        label=entity.grounding_query,
                        source="benchmark_mask:edit_region",
                    ),
                    sample.mask,
                )
        elif sample.benchmark == "CompBench":
            task = str(sample.metadata.get("task", "unknown"))
            if task not in {"add", "remove", "replace"}:
                raise AnnotationError(f"compbench_mask_only_unsupported_task:{task}")
            if sample.mask is None or sample.mask.getbbox() is None:
                raise AnnotationError(f"compbench_{task}_mask_missing_or_empty")
            parsed = self.parser.parse(
                parser_instruction,
                image=sample.image,
                mask=sample.mask,
                masked_reference_only=True,
            )
            entities = [e for e in parsed.entities if e.present_in_source]
            if len(entities) != 1:
                raise AnnotationError(
                    f"compbench_{task}_expected_one_mask_reference:got={len(entities)}"
                )
            entity = entities[0]
            start, end, _ = locate_spans(parser_instruction, [entity])[0]
            bbox = mask_bbox(sample.mask, sample.image.size)
            if bbox is None:
                raise AnnotationError(f"compbench_{task}_mask_missing_or_empty")
            component_regions[(start, end)] = (
                Region(
                    bbox_xyxy=bbox,
                    image_size=sample.image.size,
                    confidence=1.0,
                    label=entity.grounding_query,
                    source="benchmark_mask:edit_region",
                ),
                sample.mask,
            )
            entities = [entity]
            mask_guidance_used = True
            mask_guidance_reasons.append(f"compbench_{task}_edit_mask")
        else:
            parsed = self.parser.parse(parser_instruction)
            entities = [e for e in parsed.entities if e.present_in_source]
        if not entities:
            raise NoReferenceError(
                "no_source_visible_reference",
                parse_attempts=parse_attempts if masked_reference_only else None,
            )
        if masked_reference_only and len(entities) != 1:
            raise AnnotationError(
                f"refedit_expected_one_mask_reference:got_{len(entities)}"
            )
        sample.visible_reference_count = len(entities)

        located = (
            _locate_spans_allow_same(parser_instruction, entities)
            if multi_object_add or multi_object_remove or mira_bench_reference_only
            else locate_spans(parser_instruction, entities)
        )
        mixed = replace_spans(
            parser_instruction,
            located,
            spatial_destination=mask_guided_spatial_reference,
        )
        target_count = sum(e.role == "target" for _, _, e in located)

        references: list[Reference] = []
        component_region_indices: dict[tuple[int, int], int] = {}
        for i, (start, end, entity) in enumerate(located, 1):
            component_region = component_regions.get((start, end))
            if multi_object_add or multi_object_remove or mira_bench_reference_only:
                key = (start, end)
                queue = component_region_queues.get(key, [])
                queue_index = component_region_indices.get(key, 0)
                if queue_index < len(queue):
                    component_region = queue[queue_index]
                    component_region_indices[key] = queue_index + 1
            if component_region is None:
                if sample.benchmark == "CompBench":
                    raise AnnotationError(
                        f"compbench_reference_without_mask:e{i}:{entity.grounding_query!r}"
                    )
                region = self.grounder.ground(sample, entity, target_count)
            else:
                region, component_mask = component_region
                sample.reference_masks[f"e{i}"] = component_mask
            if region is None:
                raise AnnotationError(
                    f"grounding_failed:e{i}:{entity.grounding_query!r}"
                )
            if region.confidence < self.min_confidence:
                raise AnnotationError(
                    f"grounding_low_confidence:e{i}:{region.confidence:.3f}"
                )
            references.append(
                Reference(
                    token=f"<v{i}>",
                    entity_id=f"e{i}",
                    span=entity.span,
                    start=start,
                    end=end,
                    role=entity.role,
                    grounding_query=entity.grounding_query,
                    region=region,
                )
            )

        record = AnnotationRecord(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            split=sample.split,
            original_instruction=sample.instruction,
            parser_instruction=parser_instruction,
            mixed_instruction=mixed,
            references=references,
            source_ref=sample.source_ref,
            target_ref=sample.target_ref,
            metadata={
                **sample.metadata,
                **(
                    {
                        "mask_guidance_used": mask_guidance_used,
                        "mask_guidance_reasons": mask_guidance_reasons,
                    }
                    if (
                        masked_reference_only
                        or mask_guided_spatial_reference
                        or multi_object_add
                        or multi_object_remove
                        or mira_bench_reference_only
                    )
                    else {}
                ),
            },
            linguistic_stats=linguistic_stats(sample.instruction, mixed),
        )
        record.validation = validate_record(record, mixed)
        if not record.validation["pass"]:
            raise AnnotationError("validation_failed")
        return record
