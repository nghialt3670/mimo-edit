from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from framework.inference.config import KleinInferenceConfig
from framework.inference.flux2_klein import flux2_klein_add_object_with_mask


def _pool(pipe=None, **config):
    return SimpleNamespace(
        flux2_klein_pipeline=pipe or object(),
        config=SimpleNamespace(klein=KleinInferenceConfig(**config)),
        pctnet_pipeline=object(),
    )


def _red_square():
    edited = Image.new("RGB", (48, 48), "black")
    edited.paste((255, 0, 0), (10, 10, 40, 40))
    return edited


class Flux2KleinAddTests(unittest.TestCase):
    def test_mask_add_resizes_slightly_mismatched_model_output(self):
        image = Image.new("RGB", (200, 160), "black")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (50, 40, 80, 70))
        edited = Image.new("RGB", (48, 48), "black")
        edited.paste((255, 0, 0), (10, 10, 40, 40))  # mask area inside the crop

        with (
            patch(
                "framework.inference.flux2_klein.add_object.generate_klein_edit",
                return_value=edited,
            ),
            patch(
                "framework.inference.flux2_klein.utils.harmonize_with_mask",
                side_effect=lambda _p, image, _m: image,
            ),
            patch(
                "framework.inference.flux2_klein.add_object.segment_added_object",
                side_effect=lambda _pool, _before, _after, region, *_a, **_k: region,
            ),
            patch(
                "framework.inference.flux2_klein.utils.vitmatte_alpha",
                side_effect=lambda _pool, _img, m: m,
            ),
        ):
            result = flux2_klein_add_object_with_mask(
                _pool(mask_context_padding=10),
                image,
                mask,
                "a red object",
            )

        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((60, 50)), (255, 0, 0))
        self.assertEqual(result.getpixel((45, 50)), (0, 0, 0))
        self.assertEqual(result.getpixel((100, 100)), (0, 0, 0))

    def test_retries_once_when_nothing_was_added_then_raises(self):
        from framework.inference.flux2_klein.add_object import _generate_with_retry
        from framework.inference.flux2_klein.utils import NothingAdded

        before = Image.new("RGB", (40, 40), "gray")
        region = np.ones((40, 40), bool)
        calls = []

        def generate(attempt):
            calls.append(attempt)
            return before.copy()

        with self.assertRaises(NothingAdded):
            _generate_with_retry(None, before, region, "a dragon", generate, None)
        self.assertEqual(calls, [0, 1])

        calls.clear()
        debug = {}

        def generate_second(attempt):
            calls.append(attempt)
            out = before.copy()
            if attempt:
                out.paste("red", (10, 10, 30, 30))
            return out

        with patch("framework.inference.sam3.sam3_segment_masks", return_value=[]):
            _, mask = _generate_with_retry(
                SimpleNamespace(), before, region, "a box", generate_second, debug
            )
        self.assertEqual(debug["attempt"], 1)
        self.assertTrue(mask[20, 20])


if __name__ == "__main__":
    unittest.main()
