from __future__ import annotations

import json
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from annotation import mira_bench
from annotation.core import Annotator, NoReferenceError
from annotation.grounder import MaskOnlyGrounder
from annotation.runner import convert_rows
from annotation.schema import ParsedEntity, ParsedInstruction


class _Parser:
    def __init__(
        self,
        spans: list[str],
        *,
        first_call_empty: bool = False,
        always_empty: bool = False,
    ) -> None:
        self.spans = iter(spans)
        self.calls: list[dict[str, object]] = []
        self.first_call_empty = first_call_empty
        self.always_empty = always_empty

    def parse(
        self,
        instruction: str,
        *,
        image: Image.Image | None = None,
        mask: Image.Image | None = None,
        masked_reference_only: bool = False,
        mask_guided_spatial_reference: bool = False,
        mira_bench_masked_reference: bool = False,
    ) -> ParsedInstruction:
        self.calls.append(
            {
                "instruction": instruction,
                "image": image,
                "mask": mask,
                "masked_reference_only": masked_reference_only,
                "mask_guided_spatial_reference": mask_guided_spatial_reference,
                "mira_bench_masked_reference": mira_bench_masked_reference,
            }
        )
        if self.always_empty or self.first_call_empty:
            self.first_call_empty = False
            return ParsedInstruction(operation="edit")
        span = next(self.spans)
        return ParsedInstruction(
            operation="edit",
            entities=[
                ParsedEntity(
                    span=span,
                    role="target",
                    grounding_query=span,
                )
            ],
        )


def _row() -> dict:
    return {
        "image": "0.jpg",
        "editing_instruction": (
            "Turn the left star blue, make the upper right star red, "
            "shrink the lower left star, rotate the lower right star, "
            "and remove the center star."
        ),
        "mask": [
            [1, 1, 6, 1, 6, 6, 1, 6],
            [10, 1, 15, 1, 15, 6, 10, 6],
            [[1, 10, 6, 10, 6, 15, 1, 15]],
            [10, 10, 15, 10, 15, 15, 10, 15],
            [20, 20, 25, 20, 25, 25, 20, 25],
        ],
    }


class MiraBenchAnnotationTests(unittest.TestCase):
    def test_adapter_and_annotation_use_five_native_masks(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (40, 40), "white").save(root / "0.jpg")

            sample = mira_bench(_row(), "test", 0, image_root=root)
            parser = _Parser(
                [
                    "the left star",
                    "the upper right star",
                    "the lower left star",
                    "the lower right star",
                    "the center star",
                ]
            )
            result = Annotator(
                parser=parser,
                grounder=MaskOnlyGrounder(),
            ).annotate(sample)

        self.assertEqual(sample.benchmark, "MIRA-Bench")
        self.assertIsNone(sample.target)
        self.assertIsNone(sample.target_ref)
        self.assertEqual(len(result.references), 5)
        self.assertEqual(
            result.mixed_instruction,
            "Turn <v1> blue, make <v2> red, shrink <v3>, rotate <v4>, and remove <v5>.",
        )
        self.assertEqual(
            [reference.span for reference in result.references],
            [
                "the left star",
                "the upper right star",
                "the lower left star",
                "the lower right star",
                "the center star",
            ],
        )
        self.assertEqual(len(parser.calls), 5)
        self.assertTrue(
            all(
                call["masked_reference_only"]
                and not call["mira_bench_masked_reference"]
                and call["image"] is sample.image
                and call["mask"] is not None
                for call in parser.calls
            )
        )
        self.assertEqual(
            [reference.region.bbox_xyxy for reference in result.references],
            [
                (1.0, 1.0, 7.0, 7.0),
                (10.0, 1.0, 16.0, 7.0),
                (1.0, 10.0, 7.0, 16.0),
                (10.0, 10.0, 16.0, 16.0),
                (20.0, 20.0, 26.0, 26.0),
            ],
        )
        self.assertTrue(
            all(
                reference.region.source == "benchmark_mask:object_instance"
                for reference in result.references
            )
        )
        self.assertEqual(
            result.metadata["mask_guidance_reasons"], ["mira_bench_reference_masks"]
        )
        self.assertEqual(len(sample.reference_masks), 5)

    def test_convert_rows_writes_a_resumable_mira_bench_annotation(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (40, 40), "white").save(root / "0.jpg")
            row = _row()
            output = root / "run" / "mira_bench_mixed.jsonl"
            parser = _Parser(
                [
                    "the left star",
                    "the upper right star",
                    "the lower left star",
                    "the lower right star",
                    "the center star",
                ],
                first_call_empty=True,
            )
            summary = convert_rows(
                rows=[row],
                adapter=partial(mira_bench, image_root=root),
                split="test",
                annotator=Annotator(
                    parser=parser,
                    grounder=MaskOnlyGrounder(),
                ),
                output=output,
                limit=1,
            )

            saved = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(saved["benchmark"], "MIRA-Bench")
        self.assertEqual(len(saved["references"]), 5)
        self.assertTrue(parser.calls[0]["masked_reference_only"])
        self.assertFalse(parser.calls[0]["mira_bench_masked_reference"])
        self.assertTrue(parser.calls[1]["mira_bench_masked_reference"])
        self.assertEqual(len(parser.calls), 6)
        self.assertTrue(
            all(reference["region"]["mask_path"] for reference in saved["references"])
        )

    def test_adapter_rejects_mask_count_mismatches(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (40, 40), "white").save(root / "0.jpg")
            row = _row()
            row["mask"] = row["mask"][:-1]

            with self.assertRaisesRegex(ValueError, "must contain 5 polygons"):
                mira_bench(row, "test", 0, image_root=root)

    def test_unresolved_mask_is_audited_after_one_llm_retry(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (40, 40), "white").save(root / "0.jpg")
            sample = mira_bench(_row(), "test", 0, image_root=root)
            parser = _Parser([], always_empty=True)

            with self.assertRaisesRegex(
                NoReferenceError, "mira_bench_mask_region_no_source_reference:e1"
            ) as error:
                Annotator(
                    parser=parser,
                    grounder=MaskOnlyGrounder(),
                ).annotate(sample)

        self.assertEqual(
            [attempt["mode"] for attempt in error.exception.parse_attempts],
            ["masked_region_1", "masked_region_1_retry"],
        )
        self.assertEqual(len(parser.calls), 2)

    def test_repeated_llm_spans_use_distinct_instruction_occurrences(self):
        instruction = (
            "Make the left star blue, make the left star bright, "
            "shrink the lower left star, rotate the lower right star, "
            "and remove the center star."
        )
        row = _row()
        row["editing_instruction"] = instruction
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (40, 40), "white").save(root / "0.jpg")
            sample = mira_bench(row, "test", 0, image_root=root)
            parser = _Parser(
                [
                    "the left star",
                    "the left star",
                    "the lower left star",
                    "the lower right star",
                    "the center star",
                ]
            )
            result = Annotator(
                parser=parser,
                grounder=MaskOnlyGrounder(),
            ).annotate(sample)

        self.assertEqual(
            [reference.start for reference in result.references[:2]],
            [
                instruction.index("the left star"),
                instruction.rindex("the left star"),
            ],
        )
