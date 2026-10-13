from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from framework.inference.config import KleinInferenceConfig
from framework.inference.flux2_klein import flux2_klein_add_object_with_mask


class Flux2KleinAddTests(unittest.TestCase):
    def test_mask_add_resizes_slightly_mismatched_model_output(self):
        image = Image.new("RGB", (200, 160), "black")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (50, 40, 80, 70))
        edited = Image.new("RGB", (48, 48), "red")

        with patch(
            "framework.inference.flux2_klein._generate_klein_edit",
            return_value=edited,
        ):
            result = flux2_klein_add_object_with_mask(
                object(),
                image,
                mask,
                "a red object",
                config=KleinInferenceConfig(mask_context_padding=10),
            )

        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((60, 50)), (255, 0, 0))
        self.assertEqual(result.getpixel((49, 50)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
