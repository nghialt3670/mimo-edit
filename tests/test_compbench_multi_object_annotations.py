from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from annotation.core import Annotator
from annotation.grounder import mask_components
from annotation.schema import ParsedEntity, ParsedInstruction, Region, Sample


class _Parser:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict[str, object]] = []

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
        return self.response(instruction, mask)


class _Grounder:
    def ground(
        self, sample: Sample, entity: ParsedEntity, visible_target_count: int
    ) -> Region:
        raise AssertionError("CompBench references must come from benchmark masks")


def _entity(span: str, query: str = "masked object") -> ParsedEntity:
    return ParsedEntity(
        span=span,
        role="target",
        grounding_query=query,
    )


class CompBenchMultiObjectAnnotationTests(unittest.TestCase):
    def test_remove_keeps_two_masks_when_both_parse_to_collective_span(self):
        image = Image.new("RGB", (40, 40), "white")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (2, 2, 8, 8))
        mask.paste(255, (28, 28, 34, 34))
        parser = _Parser(
            lambda instruction, component: ParsedInstruction(
                operation="remove",
                entities=[_entity("the two fish", "fish")],
            )
        )
        sample = Sample(
            sample_id="multi_object_remove:two-fish.png",
            benchmark="CompBench",
            split="train",
            image=image,
            instruction="Remove the two fish.",
            mask=mask,
            metadata={
                "task": "multi_object_remove",
                "caption": "rock",
            },
        )

        result = Annotator(parser=parser, grounder=_Grounder()).annotate(sample)

        self.assertEqual(result.mixed_instruction, "Remove <v1> and <v2>.")
        self.assertEqual(len(result.references), 2)
        self.assertEqual(
            [reference.span for reference in result.references],
            ["the two fish", "the two fish"],
        )
        first = np.asarray(sample.reference_masks["e1"]) > 0
        second = np.asarray(sample.reference_masks["e2"]) > 0
        self.assertTrue(first.any())
        self.assertTrue(second.any())
        self.assertFalse(np.logical_and(first, second).any())
        self.assertEqual(len(parser.calls), 2)
        self.assertTrue(
            all(call["masked_reference_only"] for call in parser.calls)
        )

    def test_connected_two_object_mask_is_split_without_a_strong_valley(self):
        yy, xx = np.mgrid[:120, :120]
        first_object = (xx - 42) ** 2 + (yy - 60) ** 2 <= 32**2
        second_object = (xx - 78) ** 2 + (yy - 60) ** 2 <= 32**2
        source = first_object | second_object
        mask = Image.fromarray(source.astype(np.uint8) * 255)

        components = mask_components(mask, expected_count=2)

        self.assertEqual(len(components), 2)
        first = np.asarray(components[0][1]) > 0
        second = np.asarray(components[1][1]) > 0
        self.assertTrue(first.any())
        self.assertTrue(second.any())
        self.assertFalse(np.logical_and(first, second).any())
        np.testing.assert_array_equal(np.logical_or(first, second), source)
        self.assertTrue(first[60, 42] or second[60, 42])
        self.assertTrue(first[60, 78] or second[60, 78])
        self.assertNotEqual(first[60, 42], first[60, 78])

    def test_multi_add_creates_two_references_from_a_connected_mask(self):
        image = Image.new("RGB", (120, 120), "white")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (10, 10, 110, 110))

        def parse_by_mask_position(instruction, component):
            _, y0, _, y1 = component.getbbox()
            destination = "up" if (y0 + y1) / 2 < 60 else "down"
            return ParsedInstruction(
                operation="add",
                entities=[_entity(destination, "destination")],
            )

        parser = _Parser(parse_by_mask_position)
        sample = Sample(
            sample_id="multi_object_add:touching-pair.png",
            benchmark="CompBench",
            split="train",
            image=image,
            instruction="Add two white rabbits up and down.",
            mask=mask,
            metadata={"task": "multi_object_add", "caption": "rabbit|rabbit"},
        )

        result = Annotator(parser=parser, grounder=_Grounder()).annotate(sample)

        self.assertEqual(
            result.mixed_instruction, "Add two white rabbits <v1> and <v2>."
        )
        self.assertEqual(
            [reference.span for reference in result.references], ["up", "down"]
        )
        self.assertEqual(len(parser.calls), 2)

    def test_multi_add_preserves_two_masks_for_one_shared_destination(self):
        image = Image.new("RGB", (120, 120), "white")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (10, 10, 110, 110))
        parser = _Parser(
            lambda instruction, component: ParsedInstruction(
                operation="add",
                entities=[_entity("the upper left corner", "destination")],
            )
        )
        sample = Sample(
            sample_id="multi_object_add:shared-place.png",
            benchmark="CompBench",
            split="train",
            image=image,
            instruction="Add two white rabbits in the upper left corner.",
            mask=mask,
            metadata={"task": "multi_object_add", "caption": "rabbit|rabbit"},
        )

        result = Annotator(parser=parser, grounder=_Grounder()).annotate(sample)

        self.assertEqual(
            result.mixed_instruction,
            "Add two white rabbits in <v1> and <v2>.",
        )
        self.assertEqual(len(result.references), 2)
        self.assertEqual(len(parser.calls), 2)
        first = np.asarray(sample.reference_masks["e1"]) > 0
        second = np.asarray(sample.reference_masks["e2"]) > 0
        self.assertFalse(np.logical_and(first, second).any())

    def test_multi_add_resolves_overlapping_spans_with_extra_leading_preposition(self):
        image = Image.new("RGB", (40, 40), "white")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (2, 2, 18, 18))
        mask.paste(255, (22, 22, 38, 38))
        responses = iter(
            (
                "on the leftmost",
                "the leftmost",
            )
        )
        parser = _Parser(
            lambda instruction, component: ParsedInstruction(
                operation="add",
                entities=[_entity(next(responses), "destination")],
            )
        )
        sample = Sample(
            sample_id="multi_object_add:overlapping-preposition.png",
            benchmark="CompBench",
            split="train",
            image=image,
            instruction="Add two chickens on the leftmost.",
            mask=mask,
            metadata={"task": "multi_object_add", "caption": "chicken|chicken"},
        )

        result = Annotator(parser=parser, grounder=_Grounder()).annotate(sample)

        self.assertEqual(
            result.mixed_instruction,
            "Add two chickens on <v1> and <v2>.",
        )
        self.assertEqual([ref.span for ref in result.references], ["the leftmost"] * 2)

    def test_multi_add_trims_conjoined_non_location_tail(self):
        image = Image.new("RGB", (40, 40), "white")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (2, 2, 18, 18))
        mask.paste(255, (22, 22, 38, 38))
        responses = iter(
            (
                "on the left of the black dog",
                "on the left of the black dog and a woman wearing white top and blue jeans",
            )
        )
        parser = _Parser(
            lambda instruction, component: ParsedInstruction(
                operation="add",
                entities=[_entity(next(responses), "destination")],
            )
        )
        instruction = (
            "Add a yellow dog on the left of the black dog and a woman wearing "
            "white top and blue jeans."
        )
        sample = Sample(
            sample_id="multi_object_add:overlapping-conjoined-tail.png",
            benchmark="CompBench",
            split="train",
            image=image,
            instruction=instruction,
            mask=mask,
            metadata={"task": "multi_object_add", "caption": "dog|woman"},
        )

        result = Annotator(parser=parser, grounder=_Grounder()).annotate(sample)

        self.assertEqual(
            result.mixed_instruction,
            "Add a yellow dog at <v1> and <v2> and a woman wearing "
            "white top and blue jeans.",
        )
        self.assertEqual(
            [ref.span for ref in result.references],
            ["on the left of the black dog"] * 2,
        )

    def test_remove_uses_caption_count_when_instruction_is_plural(self):
        image = Image.new("RGB", (40, 40), "white")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (2, 2, 18, 18))
        mask.paste(255, (22, 22, 38, 38))
        parser = _Parser(
            lambda instruction, component: ParsedInstruction(
                operation="remove",
                entities=[_entity("the rabbits", "rabbit")],
            )
        )
        sample = Sample(
            sample_id="multi_object_remove:plural-caption-count.png",
            benchmark="CompBench",
            split="train",
            image=image,
            instruction="Remove the rabbits.",
            mask=mask,
            metadata={
                "task": "multi_object_remove",
                "caption": "a rabbit|a rabbit",
            },
        )

        result = Annotator(parser=parser, grounder=_Grounder()).annotate(sample)

        self.assertEqual(result.mixed_instruction, "Remove <v1> and <v2>.")
        self.assertEqual(len(result.references), 2)


if __name__ == "__main__":
    unittest.main()
