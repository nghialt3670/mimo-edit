from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch
from PIL import Image

from framework.core.agent import _SYSTEM_PROMPT
from framework.inference.grounding_dino import grounding_dino_segment_region
from framework.tools import registry


class GroundingDinoSegmentRegionTests(unittest.TestCase):
    def setUp(self):
        class Inputs(dict):
            input_ids = torch.tensor([[1]])

            def to(self, device):
                return self

        self.inputs = Inputs()
        self.model = Mock()
        self.processor = Mock()
        self.processor.return_value = self.inputs
        self.pool = SimpleNamespace(
            device="cpu",
            config=SimpleNamespace(
                grounding_dino=SimpleNamespace(
                    box_threshold=0.35,
                    text_threshold=0.25,
                )
            ),
            grounding_dino_pipelines=SimpleNamespace(
                model=self.model,
                processor=self.processor,
            ),
        )
        self.image = Image.new("RGB", (100, 80))

    def test_grounds_best_box_and_segments_it_with_sam3(self):
        self.processor.post_process_grounded_object_detection.return_value = [
            {
                "scores": torch.tensor([0.4, 0.9]),
                "boxes": torch.tensor(
                    [
                        [2.0, 3.0, 30.0, 40.0],
                        [-5.0, 10.0, 110.0, 70.0],
                    ]
                ),
            }
        ]
        expected_mask = Image.new("L", self.image.size, 255)

        with patch(
            "framework.inference.grounding_dino.sam3_segment_mask",
            return_value=expected_mask,
        ) as segment:
            result = grounding_dino_segment_region(
                self.pool, self.image, "  the car. "
            )

        self.assertIs(result, expected_mask)
        self.processor.assert_called_once_with(
            images=self.image,
            text=[["the car"]],
            return_tensors="pt",
        )
        self.processor.post_process_grounded_object_detection.assert_called_once_with(
            ANY,
            self.inputs.input_ids,
            threshold=0.35,
            text_threshold=0.25,
            target_sizes=[(80, 100)],
            text_labels=[["the car"]],
        )
        segment.assert_called_once_with(
            self.pool,
            self.image,
            box=[0.0, 10.0, 100.0, 70.0],
        )

    def test_raises_when_no_detection_matches(self):
        self.processor.post_process_grounded_object_detection.return_value = [
            {"scores": torch.tensor([]), "boxes": torch.empty((0, 4))}
        ]

        with self.assertRaisesRegex(ValueError, "No image region matched"):
            grounding_dino_segment_region(self.pool, self.image, "a car")

    def test_rejects_empty_description(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            grounding_dino_segment_region(self.pool, self.image, " ")

    def test_segment_region_is_registered_and_agent_prompt_is_mode_agnostic(self):
        tool_names = {tool.name for tool in registry.build()}

        self.assertIn("segment_region", tool_names)
        self.assertNotIn("mixed visual-prompt mode", _SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
