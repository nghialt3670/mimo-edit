import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from framework.inference.config import KleinInferenceConfig
from framework.inference.flux2_klein import flux2_klein_edit_object


class EditRegionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (240, 240), (100, 100, 100))
        self.mask = Image.new("L", self.image.size, 0)
        self.mask.paste(255, (100, 100, 140, 140))
        self.pool = SimpleNamespace(
            flux2_klein_pipeline=object(),
            config=SimpleNamespace(klein=KleinInferenceConfig()),
            pctnet_pipeline=object(),
        )

    def _edit(self, paint, **kwargs):
        crops = []

        def fake(_pipeline, crop, _prompt, _config):
            crops.append(crop.size)
            return paint(crop)

        with patch(
            "framework.inference.flux2_klein.edit_object.generate_klein_edit", fake
        ):
            result = flux2_klein_edit_object(
                self.pool,
                self.image,
                self.mask,
                "x",
                harmonize=False,
                matting=False,
                **kwargs,
            )
        return result, crops

    def test_unchanged_klein_output_keeps_image(self):
        result, _ = self._edit(
            lambda crop: Image.new("RGB", crop.size, (100, 100, 100)), mode="local"
        )
        self.assertEqual(result.getpixel((120, 120)), (100, 100, 100))

    def test_mask_is_pasted_even_when_klein_barely_changes_it(self):
        result, _ = self._edit(
            lambda crop: Image.new("RGB", crop.size, (103, 100, 100)), mode="local"
        )
        self.assertEqual(result.getpixel((120, 120)), (103, 100, 100))

    def test_regional_change_touching_mask_outside_it_is_pasted(self):
        def paint(crop):
            # Crop origin is 100 - 20 (regional margin) - 6 (feather support) = 74.
            out = Image.new("RGB", crop.size, (100, 100, 100))
            out.paste((30, 30, 200), (100 - 74, 100 - 74, 160 - 74, 140 - 74))
            return out

        result, _ = self._edit(paint, mode="regional", padding=0.5)
        self.assertEqual(result.getpixel((120, 120)), (30, 30, 200))
        self.assertEqual(result.getpixel((150, 120)), (30, 30, 200))
        self.assertEqual(result.getpixel((200, 120)), (100, 100, 100))

    def test_local_change_far_from_mask_is_not_pasted(self):
        def paint(crop):
            # Crop origin is 100 - 16 (local context) - 6 (feather support) = 78.
            out = Image.new("RGB", crop.size, (100, 100, 100))
            out.paste((30, 30, 200), (100 - 78, 100 - 78, 160 - 78, 140 - 78))
            return out

        result, _ = self._edit(paint, mode="local")
        self.assertEqual(result.getpixel((120, 120)), (30, 30, 200))
        self.assertEqual(result.getpixel((156, 120)), (100, 100, 100))

    def test_distant_unconnected_change_is_ignored(self):
        def paint(crop):
            out = Image.new("RGB", crop.size, (100, 100, 100))
            out.paste((255, 0, 0), (0, 0, 8, 8))  # far corner, not touching the mask
            out.paste((255, 255, 255), (30, 30, 70, 70))
            return out

        result, _ = self._edit(paint, mode="local")
        self.assertEqual(result.getpixel((74, 74)), (100, 100, 100))

    def test_global_returns_full_klein_output(self):
        result, crops = self._edit(
            lambda crop: Image.new("RGB", crop.size, "red"), mode="global"
        )
        self.assertEqual(crops, [(240, 240)])
        self.assertEqual(result.getpixel((5, 5)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
