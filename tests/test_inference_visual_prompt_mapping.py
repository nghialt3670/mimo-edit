from __future__ import annotations

import unittest

from PIL import Image

from evaluation.common import (
    build_agent_context,
    build_agent_task,
    load_sample,
    select_reference_prompts,
)
from evaluation.spatial import needs_spatial_anchor_box


class InferenceVisualPromptMappingTests(unittest.TestCase):
    def test_legacy_region_bbox_is_bound_when_visual_prompts_are_missing(self):
        class ImageLoader:
            @staticmethod
            def load_pair(record):
                return Image.new("RGB", (512, 512)), None

        record = {
            "sample_id": "easy:000",
            "mixed_instruction": "Change the color of <v1> to white.",
            "references": [
                {
                    "token": "<v1>",
                    "span": "the boat with bicycles and people",
                    "region": {
                        "bbox_xyxy": [40.0, 218.0, 365.0, 299.0],
                        "image_size": [512, 512],
                    },
                }
            ],
        }

        sample = load_sample(
            record,
            benchmark="refedit_bench",
            image_loader=ImageLoader(),
        )
        select_reference_prompts(sample)
        task, objects = build_agent_task(sample)

        self.assertEqual(task, "Change the color of <box_0> to white.")
        self.assertEqual(objects["box_0"], [40.0, 218.0, 365.0, 299.0])
        self.assertNotIn("<v1>", task)
        self.assertIn("box_0", build_agent_context(sample))

    def test_mixed_instruction_names_exact_bound_prompt_variable(self):
        sample = {
            "sample_id": "easy:003",
            "instruction": "Remove <v1>.",
            "source_image": Image.new("RGB", (8, 8)),
            "references": [
                {
                    "token": "<v1>",
                    "span": "the surfboard",
                    "bbox_xyxy": None,
                    "point_xy": [3.0, 4.0],
                    "scribble_strokes": [],
                    "spatial_destination": False,
                }
            ],
        }

        select_reference_prompts(sample)
        task, objects = build_agent_task(sample)
        context = build_agent_context(sample)

        self.assertEqual(task, "Remove <point_0>.")
        self.assertIn("point_0", objects)
        self.assertNotIn("<v1>", task)
        self.assertIn("<point_0>", context)
        self.assertNotIn("surfboard", context)
        self.assertEqual(sample["references"][0]["display_token"], "<point_0>")

    def test_mixed_spatial_destination_keeps_placement_description(self):
        sample = {
            "instruction": "Add a bridge at <box_0>.",
            "references": [
                {
                    "token": "<v1>",
                    "display_token": "<box_0>",
                    "span": "below the mountain",
                    "selected_prompt": {"type": "box", "variable": "box_0"},
                    "spatial_destination": True,
                }
            ],
        }

        context = build_agent_context(sample)

        self.assertIn("below the mountain", context)

    def test_spatial_placement_guides_manual_destination_box_calculation(self):
        sample = {
            "instruction": "Add a bird above <v1>.",
            "references": [
                {
                    "token": "<v1>",
                    "display_token": "<box_0>",
                    "span": "the tree",
                    "selected_prompt": {"type": "box", "variable": "box_0"},
                    "spatial_destination": False,
                }
            ],
        }

        context = build_agent_context(sample)

        self.assertTrue(needs_spatial_anchor_box(sample["instruction"]))
        self.assertIn("calculate an approximate destination", context)
        self.assertIn("add_object(box=...)", context)
        self.assertIn("full-object anchor", context)

    def test_detects_diagonal_and_cardinal_relative_placement(self):
        for instruction in (
            "Place a lamp to the right of the sofa.",
            "Add a plant below the window.",
            "Put a picture in the top-left of the wall.",
            "Position a chair upper right of the desk.",
        ):
            with self.subTest(instruction=instruction):
                self.assertTrue(needs_spatial_anchor_box(instruction))

    def test_text_mode_keeps_instruction_without_visual_prompt_tokens(self):
        sample = {
            "sample_id": "easy:003",
            "instruction": "Remove the surfboard.",
            "references": [
                {
                    "token": "<v1>",
                    "span": "the surfboard",
                    "bbox_xyxy": [1, 1, 7, 7],
                    "point_xy": [3.0, 4.0],
                    "scribble_strokes": [],
                    "spatial_destination": False,
                }
            ],
        }

        select_reference_prompts(sample, prompt_mode="text")

        self.assertEqual(sample["instruction"], "Remove the surfboard.")
        self.assertIsNone(sample["references"][0]["selected_prompt"])

    def test_agent_context_does_not_expose_inference_prompt_family(self):
        text_context = build_agent_context(
            {
                "instruction": "Remove the surfboard.",
                "references": [
                    {
                        "token": "<v1>",
                        "span": "the surfboard",
                        "selected_prompt": None,
                    }
                ],
            },
            prompt_mode="text",
        )
        visual_context = build_agent_context(
            {
                "instruction": "Remove <point_0>.",
                "references": [
                    {
                        "token": "<v1>",
                        "display_token": "<point_0>",
                        "span": "the surfboard",
                        "selected_prompt": {
                            "type": "point",
                            "variable": "point_0",
                        },
                        "spatial_destination": False,
                    }
                ],
            },
            prompt_mode="mixed",
        )

        for context in (text_context, visual_context):
            with self.subTest(context=context):
                lowered_context = context.lower()
                self.assertNotIn("text mode", lowered_context)
                self.assertNotIn("mixed mode", lowered_context)
                self.assertNotIn("text-only", lowered_context)
        self.assertIn("No visual selector is supplied", text_context)
        self.assertIn("exactly matching variable", visual_context)


if __name__ == "__main__":
    unittest.main()
