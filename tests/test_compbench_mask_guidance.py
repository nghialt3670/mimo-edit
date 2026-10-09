from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from annotation.core import Annotator
from annotation.grounder import HybridGrounder, MaskOnlyGrounder, mask_components
from annotation.runner import convert_rows
from annotation.schema import ParsedEntity, ParsedInstruction, Region, Sample


class SpatialParser:
    def __init__(self, span: str | list[str] | None = "the leftmost") -> None:
        self.calls: list[dict[str, object]] = []
        self.spans = [span] if isinstance(span, str) or span is None else span

    def parse(
        self,
        instruction: str,
        *,
        image: Image.Image | None = None,
        mask: Image.Image | None = None,
        masked_reference_only: bool = False,
        mask_guided_spatial_reference: bool = False,
    ) -> ParsedInstruction:
        self.calls.append(
            {
                "instruction": instruction,
                "image": image,
                "mask": mask,
                "masked_reference_only": masked_reference_only,
                "mask_guided_spatial_reference": mask_guided_spatial_reference,
            }
        )
        span = self.spans[min(len(self.calls) - 1, len(self.spans) - 1)]
        if span is None:
            return ParsedInstruction(operation="add")
        return ParsedInstruction(
            operation="add",
            entities=[
                ParsedEntity(
                    span=span,
                    role="target",
                    grounding_query="masked addition destination",
                )
            ],
        )


class UnexpectedFallback:
    def ground(self, sample: Sample, entity: ParsedEntity, visible_target_count: int) -> Region | None:
        _ = sample, visible_target_count
        raise AssertionError(f"Unexpected text grounding for {entity.grounding_query!r}")


def _sample() -> Sample:
    image = Image.new("RGB", (100, 80), "white")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((4, 10, 30, 45), fill=255)
    return Sample(
        sample_id="add:sample",
        benchmark="CompBench",
        split="train",
        image=image,
        instruction="add a gray fish on the leftmost with its head downward",
        mask=mask,
        metadata={"task": "add"},
    )


class CompBenchMaskGuidanceTests(unittest.TestCase):
    def test_add_uses_masked_spatial_phrase_and_mask_bbox(self) -> None:
        sample = _sample()
        parser = SpatialParser()
        annotator = Annotator(
            parser=parser,
            grounder=MaskOnlyGrounder(),
        )

        record = annotator.annotate(sample)

        self.assertEqual(len(parser.calls), 1)
        self.assertIs(parser.calls[0]["image"], sample.image)
        self.assertIs(parser.calls[0]["mask"], sample.mask)
        self.assertTrue(parser.calls[0]["mask_guided_spatial_reference"])
        self.assertFalse(parser.calls[0]["masked_reference_only"])
        self.assertEqual(record.mixed_instruction, "add a gray fish on <v1> with its head downward")
        self.assertEqual(record.references[0].region.bbox_xyxy, (4.0, 10.0, 31.0, 46.0))
        self.assertEqual(record.references[0].region.source, "benchmark_mask:edit_region")
        self.assertTrue(record.metadata["mask_guidance_used"])

    def test_multi_object_add_uses_one_reference_per_mask_component(self) -> None:
        sample = _sample()
        sample.metadata["task"] = "multi_object_add"
        sample.metadata["caption"] = "a chicken|a chicken"
        sample.instruction = (
            "add a chicken on the bottom left and another one on the bottom right"
        )
        ImageDraw.Draw(sample.mask).rectangle((70, 20, 90, 60), fill=255)
        parser = SpatialParser(span=["on the bottom left", "on the bottom right"])
        annotator = Annotator(
            parser=parser,
            grounder=MaskOnlyGrounder(),
        )

        record = annotator.annotate(sample)

        self.assertEqual(len(parser.calls), 2)
        self.assertTrue(all(call["mask_guided_spatial_reference"] for call in parser.calls))
        self.assertEqual(
            [call["mask"].getbbox() for call in parser.calls],
            [(4, 10, 31, 46), (70, 20, 91, 61)],
        )
        self.assertEqual(
            record.mixed_instruction,
            "add a chicken at <v1> and another one at <v2>",
        )
        self.assertEqual(len(record.references), 2)
        self.assertEqual(
            [ref.region.bbox_xyxy for ref in record.references],
            [(4.0, 10.0, 31.0, 46.0), (70.0, 20.0, 91.0, 61.0)],
        )
        self.assertEqual(sample.reference_masks["e1"].getbbox(), (4, 10, 31, 46))
        self.assertEqual(sample.reference_masks["e2"].getbbox(), (70, 20, 91, 61))
        self.assertTrue(record.metadata["mask_guidance_used"])

    def test_multi_object_add_splits_touching_mask_placements(self) -> None:
        sample = _sample()
        sample.metadata["task"] = "multi_object_add"
        sample.metadata["caption"] = "a fish|a fish"
        sample.instruction = "add a fish on the left and another fish on the right"
        mask_draw = ImageDraw.Draw(sample.mask)
        mask_draw.rectangle((4, 10, 43, 55), fill=255)
        mask_draw.rectangle((50, 15, 90, 65), fill=255)
        mask_draw.rectangle((43, 31, 51, 34), fill=255)
        foreground_pixels = sum(sample.mask.histogram()[1:])

        components = mask_components(sample.mask, expected_count=2)

        self.assertEqual(len(components), 2)
        self.assertEqual(sum(area for area, _ in components), foreground_pixels)
        bboxes = sorted(component.getbbox() for _, component in components)
        self.assertEqual(bboxes, [(4, 10, 48, 56), (48, 15, 91, 66)])

        parser = SpatialParser(span=["on the left", "on the right"])
        annotator = Annotator(
            parser=parser,
            grounder=MaskOnlyGrounder(),
        )
        record = annotator.annotate(sample)

        self.assertEqual(
            record.mixed_instruction,
            "add a fish at <v1> and another fish at <v2>",
        )
        self.assertEqual(len(record.references), 2)

    def test_multi_object_add_coalesces_repeated_location_phrase(self) -> None:
        sample = _sample()
        sample.metadata["task"] = "multi_object_add"
        sample.metadata["caption"] = "a bird|a bird"
        sample.instruction = "add two black birds on the right"
        ImageDraw.Draw(sample.mask).rectangle((4, 10, 30, 45), fill=255)
        ImageDraw.Draw(sample.mask).rectangle((70, 20, 90, 60), fill=255)
        parser = SpatialParser(span="on the right")
        annotator = Annotator(
            parser=parser,
            grounder=HybridGrounder(fallback=UnexpectedFallback()),
        )

        record = annotator.annotate(sample)

        self.assertEqual(len(parser.calls), 2)
        self.assertEqual(len(record.references), 1)
        self.assertEqual(record.references[0].span, "on the right")
        self.assertEqual(record.mixed_instruction, "add two black birds at <v1>")
        self.assertEqual(
            record.references[0].region.bbox_xyxy,
            (4.0, 10.0, 91.0, 61.0),
        )
        self.assertEqual(
            sample.reference_masks["e1"].getbbox(),
            (4, 10, 91, 61),
        )

    def test_multi_object_add_without_location_appends_destination_reference(self) -> None:
        sample = _sample()
        sample.metadata["task"] = "multi_object_add"
        sample.metadata["caption"] = "a goat|a goat"
        sample.instruction = "add two white goats"
        parser = SpatialParser(span=None)
        annotator = Annotator(
            parser=parser,
            grounder=HybridGrounder(fallback=UnexpectedFallback()),
        )

        record = annotator.annotate(sample)

        self.assertEqual(record.mixed_instruction, "add two white goats at <v1>")
        self.assertEqual(len(record.references), 1)
        self.assertEqual(record.references[0].span, "")
        self.assertEqual(record.references[0].region.bbox_xyxy, (4.0, 10.0, 31.0, 46.0))

    def test_multi_object_remove_uses_masks_without_grounding_model(self) -> None:
        sample = _sample()
        sample.metadata["task"] = "multi_object_remove"
        sample.metadata["caption"] = "a white duck|a black and white duck"
        sample.instruction = (
            "remove the white duck on the upper left and a black and white duck "
            "near the feeding bowl"
        )
        ImageDraw.Draw(sample.mask).rectangle((4, 10, 30, 45), fill=255)
        ImageDraw.Draw(sample.mask).rectangle((70, 20, 90, 60), fill=255)
        parser = SpatialParser(
            span=[
                "the white duck on the upper left",
                "a black and white duck near the feeding bowl",
            ]
        )
        annotator = Annotator(
            parser=parser,
            grounder=UnexpectedFallback(),
        )

        record = annotator.annotate(sample)

        self.assertEqual(len(parser.calls), 2)
        self.assertTrue(all(call["masked_reference_only"] for call in parser.calls))
        self.assertFalse(any(call["mask_guided_spatial_reference"] for call in parser.calls))
        self.assertEqual(
            [call["mask"].getbbox() for call in parser.calls],
            [(4, 10, 31, 46), (70, 20, 91, 61)],
        )
        self.assertEqual(
            record.mixed_instruction,
            "remove <v1> and <v2>",
        )
        self.assertEqual(
            [ref.region.bbox_xyxy for ref in record.references],
            [(4.0, 10.0, 31.0, 46.0), (70.0, 20.0, 91.0, 61.0)],
        )
        self.assertEqual(
            [ref.region.source for ref in record.references],
            ["benchmark_mask:edit_region_component"] * 2,
        )
        self.assertEqual(sample.reference_masks["e1"].getbbox(), (4, 10, 31, 46))
        self.assertEqual(sample.reference_masks["e2"].getbbox(), (70, 20, 91, 61))
        self.assertEqual(
            record.metadata["mask_guidance_reasons"],
            ["compbench_multi_object_remove_mask_regions"],
        )

    def test_multi_object_remove_trims_nested_conjoined_mask_spans(self) -> None:
        sample = _sample()
        sample.metadata["task"] = "multi_object_remove"
        sample.metadata["caption"] = "a fish|a fish"
        sample.instruction = "remove the rightmost and second rightmost fish"
        ImageDraw.Draw(sample.mask).rectangle((4, 10, 30, 45), fill=255)
        ImageDraw.Draw(sample.mask).rectangle((70, 20, 90, 60), fill=255)
        parser = SpatialParser(
            span=[
                "the rightmost and second rightmost fish",
                "second rightmost fish",
            ]
        )
        annotator = Annotator(
            parser=parser,
            grounder=MaskOnlyGrounder(),
        )

        record = annotator.annotate(sample)

        self.assertEqual(
            [reference.span for reference in record.references],
            ["the rightmost", "second rightmost fish"],
        )
        self.assertEqual(record.mixed_instruction, "remove <v1> and <v2>")
        self.assertEqual(
            [reference.region.bbox_xyxy for reference in record.references],
            [(4.0, 10.0, 31.0, 46.0), (70.0, 20.0, 91.0, 61.0)],
        )

    def test_remove_and_replace_use_dataset_masks_only(self) -> None:
        cases = [
            ("remove", "remove the red cup", "the red cup"),
            ("replace", "replace the red cup with a blue one", "the red cup"),
        ]
        for task, instruction, span in cases:
            with self.subTest(task=task):
                sample = _sample()
                sample.metadata["task"] = task
                sample.instruction = instruction
                parser = SpatialParser(span=span)
                annotator = Annotator(
                    parser=parser,
                    grounder=MaskOnlyGrounder(),
                )

                record = annotator.annotate(sample)

                self.assertTrue(parser.calls[0]["masked_reference_only"])
                self.assertEqual(len(record.references), 1)
                self.assertEqual(record.references[0].span, span)
                self.assertEqual(
                    record.references[0].region.bbox_xyxy,
                    (4.0, 10.0, 31.0, 46.0),
                )
                self.assertEqual(
                    record.references[0].region.source,
                    "benchmark_mask:edit_region",
                )
                self.assertEqual(
                    sample.reference_masks["e1"].getbbox(),
                    (4, 10, 31, 46),
                )

    def test_leading_relation_is_replaced_with_connected_destination(self) -> None:
        sample = _sample()
        sample.instruction = (
            "add a bird similar to the other two flying birds "
            "in the middle of two flying birds"
        )
        parser = SpatialParser(span="in the middle of two flying birds")
        annotator = Annotator(
            parser=parser,
            grounder=HybridGrounder(fallback=UnexpectedFallback()),
        )

        record = annotator.annotate(sample)

        self.assertEqual(
            record.mixed_instruction,
            "add a bird similar to the other two flying birds at <v1>",
        )
        self.assertEqual(
            record.references[0].span,
            "in the middle of two flying birds",
        )

    def test_previous_skip_is_retried(self) -> None:
        sample = _sample()
        parser = SpatialParser()
        annotator = Annotator(
            parser=parser,
            grounder=HybridGrounder(fallback=UnexpectedFallback()),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotations.jsonl"
            audit = Path(directory) / "audit.jsonl"
            audit.write_text(
                json.dumps(
                    {
                        "status": "skipped",
                        "sample_id": sample.sample_id,
                        "error": "no_source_visible_reference",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def adapter(row: dict, split: str, index: int) -> Sample:
                self.assertEqual(row, {})
                self.assertEqual(split, "train")
                self.assertEqual(index, 0)
                return sample

            counts = convert_rows(
                rows=[{}],
                adapter=adapter,
                split="train",
                annotator=annotator,
                output=output,
                failures=audit,
                save_assets=False,
            )

            self.assertEqual(counts["accepted"], 1)
            self.assertEqual(counts["retried"], 1)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
