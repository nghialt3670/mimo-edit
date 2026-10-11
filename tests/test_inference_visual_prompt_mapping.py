from __future__ import annotations

import unittest

from PIL import Image

from evaluation.common import (
    build_agent_context,
    build_agent_task,
    select_reference_prompts,
)


class InferenceVisualPromptMappingTests(unittest.TestCase):
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
        self.assertEqual(sample["references"][0]["display_token"], "<point_0>")

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


if __name__ == "__main__":
    unittest.main()
