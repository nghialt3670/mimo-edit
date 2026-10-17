from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from annotation.core import Annotator, NoReferenceError
from annotation.parser import MASKED_REFERENCE_PROMPT
from annotation.runner import convert_rows
from annotation.schema import ParsedEntity, ParsedInstruction, Region, Role, Sample


class _Parser:
    def __init__(self, entity: ParsedEntity):
        self.entity = entity
        self.calls: list[dict[str, object]] = []
        self.responses = [ParsedInstruction(operation="edit", entities=[entity])]

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
        response_index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[response_index]


class _Grounder:
    def ground(
        self, sample: Sample, entity: ParsedEntity, visible_target_count: int
    ) -> Region:
        return Region(
            bbox_xyxy=(1, 1, 9, 9),
            image_size=sample.image.size,
            confidence=1.0,
            label=entity.grounding_query,
            source="test_mask",
        )


class RefEditMaskGuidedAnnotationTests(unittest.TestCase):
    def _annotate(self, instruction: str, span: str, role: Role):
        image = Image.new("RGB", (10, 10), "white")
        mask = Image.new("L", (10, 10), 0)
        mask.paste(255, (1, 1, 9, 9))
        parser = _Parser(
            ParsedEntity(
                span=span,
                role=role,
                grounding_query="masked entity",
            )
        )
        sample = Sample(
            sample_id="easy:054",
            benchmark="refedit_bench",
            split="easy",
            image=image,
            instruction=instruction,
            mask=mask,
            metadata={"mask_kind": "object_instance"},
        )
        result = Annotator(
            parser=parser,
            grounder=_Grounder(),
            refedit_mask_from_start=True,
        ).annotate(sample)
        return result, parser, image, mask

    def test_prompt_distinguishes_masked_anchor_from_edit_target(self):
        self.assertIn(
            'with the MAN masked -> span "the man", role="anchor"',
            MASKED_REFERENCE_PROMPT,
        )
        self.assertIn(
            'with the SURFBOARD masked -> span "the surfboard", role="target"',
            MASKED_REFERENCE_PROMPT,
        )
        self.assertIn(
            "X is the existing source entity and Y is the requested replacement",
            MASKED_REFERENCE_PROMPT,
        )
        self.assertIn(
            'with the BOY masked -> span "the young boy", role="target"',
            MASKED_REFERENCE_PROMPT,
        )

    def test_frisbee_target_survives_when_person_is_masked(self):
        result, parser, image, mask = self._annotate(
            "Erase the pink frisbee the man is holding.",
            "the man",
            "anchor",
        )

        self.assertEqual(
            result.mixed_instruction, "Erase the pink frisbee <v1> is holding."
        )
        self.assertEqual(result.references[0].role, "anchor")
        self.assertEqual(len(parser.calls), 1)
        self.assertIs(parser.calls[0]["image"], image)
        self.assertIs(parser.calls[0]["mask"], mask)
        self.assertTrue(parser.calls[0]["masked_reference_only"])

    def test_edited_accessory_survives_when_person_is_masked(self):
        result, _, _, _ = self._annotate(
            "Get rid of the eyeglasses of the white-haired man.",
            "the white-haired man",
            "anchor",
        )

        self.assertEqual(
            result.mixed_instruction,
            "Get rid of the eyeglasses of <v1>.",
        )
        self.assertIn("eyeglasses", result.mixed_instruction)

    def test_directly_masked_edit_target_remains_a_target(self):
        result, _, _, _ = self._annotate(
            "Change the surfboard to red.",
            "the surfboard",
            "target",
        )

        self.assertEqual(result.mixed_instruction, "Change <v1> to red.")
        self.assertEqual(result.references[0].role, "target")

    def test_falls_back_to_unambiguous_text_parse_if_mask_retry_is_empty(self):
        image = Image.new("RGB", (10, 10), "white")
        mask = Image.new("L", (10, 10), 0)
        mask.paste(255, (1, 1, 9, 9))
        entity = ParsedEntity(
            span="the surfboard",
            role="target",
            grounding_query="surfboard",
        )
        parser = _Parser(entity)
        parser.responses = [
            ParsedInstruction(
                operation="edit",
                entities=[entity],
                mask_guidance_needed=True,
            ),
            ParsedInstruction(operation="edit", entities=[]),
        ]
        sample = Sample(
            sample_id="easy:003",
            benchmark="refedit_bench",
            split="easy",
            image=image,
            instruction="Change the surfboard to a guy.",
            mask=mask,
        )

        result = Annotator(parser=parser, grounder=_Grounder()).annotate(sample)

        self.assertEqual(result.mixed_instruction, "Change <v1> to a guy.")
        self.assertEqual(result.references[0].span, "the surfboard")
        self.assertIn(
            "masked_retry_empty_text_fallback",
            result.metadata["mask_guidance_reasons"],
        )

    def test_failure_retains_parse_attempt_details_in_audit(self):
        image = Image.new("RGB", (10, 10), "white")
        mask = Image.new("L", (10, 10), 0)
        mask.paste(255, (1, 1, 9, 9))
        absent_entity = ParsedEntity(
            span="a young girl",
            role="target",
            grounding_query="young girl",
            present_in_source=False,
        )
        parser = _Parser(absent_entity)
        parser.responses = [
            ParsedInstruction(
                operation="edit",
                entities=[absent_entity],
                mask_guidance_needed=True,
            ),
            ParsedInstruction(operation="edit", entities=[absent_entity]),
        ]
        sample = Sample(
            sample_id="easy:013",
            benchmark="refedit_bench",
            split="easy",
            image=image,
            instruction="Change the young boy to a young girl.",
            mask=mask,
        )
        annotator = Annotator(parser=parser, grounder=_Grounder())

        with self.assertRaises(NoReferenceError) as raised:
            annotator.annotate(sample)

        self.assertEqual(
            [attempt["mode"] for attempt in raised.exception.parse_attempts],
            ["text", "masked"],
        )
        self.assertEqual(
            raised.exception.parse_attempts[0]["entities"],
            [
                {
                    "span": "a young girl",
                    "role": "target",
                    "present_in_source": False,
                }
            ],
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "annotations.jsonl"
            audit = output.with_name("audit.jsonl")
            counts = convert_rows(
                rows=[{"instruction": sample.instruction}],
                adapter=lambda row, split, index: sample,
                split="test",
                annotator=annotator,
                output=output,
                failures=audit,
                save_assets=False,
            )

            audit_record = json.loads(audit.read_text().strip())
            self.assertEqual(counts["skipped"], 1)
            self.assertEqual(len(audit_record["parse_attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
