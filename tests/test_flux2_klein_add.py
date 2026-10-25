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
    def test_mask_add_pastes_edited_crop_at_working_size(self):
        image = Image.new("RGB", (200, 160), "black")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (50, 40, 80, 70))
        # Crop is (40, 30, 90, 80) -> 50x50, contained to WORK_SIZE (1024x1024).
        # The mask region maps to roughly (205, 205)-(819, 819) in working space.
        edited = Image.new("RGB", (1024, 1024), "black")
        edited.paste((255, 0, 0), (205, 205, 819, 819))

        with (
            patch(
                "framework.inference.flux2_klein.add_object._generate_klein_edit",
                return_value=edited,
            ),
            patch(
                "framework.inference.flux2_klein.utils.harmonize_with_mask",
                side_effect=lambda _p, image, _m: image,
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


if __name__ == "__main__":
    unittest.main()
