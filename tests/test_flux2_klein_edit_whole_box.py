import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from framework.inference.config import KleinInferenceConfig
from framework.inference.flux2_klein import flux2_klein_edit_object


class WholeBoxEditTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 200), "black")
        self.mask = Image.new("L", self.image.size, 0)
        self.mask.paste(255, (80, 80, 120, 120))
        self.pool = SimpleNamespace(
            flux2_klein_pipeline=object(),
            config=SimpleNamespace(klein=KleinInferenceConfig()),
            pctnet_pipeline=object(),
        )

    def _edit(self, **kwargs):
        crops = []

        def fake(_pipeline, crop, _prompt, _config):
            crops.append(crop.size)
            return Image.new("RGB", crop.size, "red")

        with patch("framework.inference.flux2_klein._generate_klein_edit", fake):
            result = flux2_klein_edit_object(
                self.pool,
                self.image,
                self.mask,
                "x",
                harmonize=False,
                **kwargs,
            )
        return result, crops

    def test_local_whole_box_pastes_entire_box_not_only_mask(self):
        mask = Image.new("L", self.image.size, 0)
        mask.paste(255, (80, 80, 160, 160))
        mask.paste(0, (110, 110, 130, 130))  # hole inside the box
        self.mask = mask
        result, crops = self._edit(mode="local", whole_box=True)
        self.assertEqual(crops, [(80, 80)])
        self.assertEqual(result.getpixel((120, 120)), (255, 0, 0))
        self.assertEqual(result.getpixel((79, 120)), (0, 0, 0))
        self.assertEqual(result.getpixel((160, 120)), (0, 0, 0))

    def test_regional_whole_box_uses_padded_box(self):
        result, crops = self._edit(mode="regional", whole_box=True, padding=0.5)
        self.assertEqual(crops, [(80, 80)])
        self.assertEqual(result.getpixel((100, 100)), (255, 0, 0))
        self.assertEqual(result.getpixel((59, 100)), (0, 0, 0))

    def test_default_keeps_mask_only_behavior(self):
        mask = Image.new("L", self.image.size, 0)
        mask.paste(255, (40, 40, 180, 180))
        mask.paste(0, (70, 70, 150, 150))  # hole inside the box
        self.mask = mask
        result, _ = self._edit(mode="local")
        self.assertEqual(result.getpixel((120, 120)), (0, 0, 0))
        self.assertEqual(result.getpixel((45, 45)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
