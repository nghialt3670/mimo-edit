from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from framework.inference.qwen_locate import qwen_locate_region


class _FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(pixel_values=torch.zeros(1))
        self.input_ids = torch.zeros((1, 3), dtype=torch.long)

    def to(self, device: str) -> _FakeInputs:
        return self


class _FakeProcessor:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def apply_chat_template(self, messages, tokenize, add_generation_prompt) -> str:
        return "PROMPT"

    def __call__(self, text, images, return_tensors) -> _FakeInputs:
        return _FakeInputs()

    def batch_decode(self, ids, skip_special_tokens) -> list[str]:
        return [self.reply]


class _FakeModel:
    def generate(self, **kwargs) -> torch.Tensor:
        return torch.zeros((1, 5), dtype=torch.long)


class QwenLocateRegionTests(unittest.TestCase):
    def _pool(self, reply: str) -> SimpleNamespace:
        return SimpleNamespace(
            device="cpu",
            config=SimpleNamespace(qwen_locate=SimpleNamespace(max_new_tokens=32)),
            qwen_locate_pipelines=SimpleNamespace(
                model=_FakeModel(),
                processor=_FakeProcessor(reply),
            ),
        )

    def test_maps_normalized_box_to_pixels(self) -> None:
        pool = self._pool("100, 200, 500, 800]}")
        image = Image.new("RGB", (1000, 500))

        box = qwen_locate_region(pool, image, "the red mug")

        self.assertEqual(box, [100.0, 100.0, 500.0, 400.0])

    def test_clamps_out_of_range_coordinates(self) -> None:
        pool = self._pool("-50, 0, 1200, 1000]}")
        image = Image.new("RGB", (200, 100))

        box = qwen_locate_region(pool, image, "the lamp")

        self.assertEqual(box, [0.0, 0.0, 200.0, 100.0])

    def test_unparsable_output_falls_back_to_full_image(self) -> None:
        pool = self._pool("I cannot find it")
        image = Image.new("RGB", (320, 240))

        box = qwen_locate_region(pool, image, "a dog")

        self.assertEqual(box, [0.0, 0.0, 320.0, 240.0])

    def test_degenerate_box_falls_back_to_full_image(self) -> None:
        pool = self._pool("500, 500, 100, 500]}")
        image = Image.new("RGB", (320, 240))

        box = qwen_locate_region(pool, image, "a cup")

        self.assertEqual(box, [0.0, 0.0, 320.0, 240.0])

    def test_empty_prompt_is_rejected(self) -> None:
        pool = self._pool("0, 0, 1000, 1000]}")
        image = Image.new("RGB", (32, 32))

        with self.assertRaisesRegex(ValueError, "prompt"):
            qwen_locate_region(pool, image, "  ")


if __name__ == "__main__":
    unittest.main()
