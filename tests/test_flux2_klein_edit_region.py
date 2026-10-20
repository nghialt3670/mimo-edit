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

        with (
            patch(
                "framework.inference.flux2_klein.edit_object.generate_klein_edit",
                fake,
            ),
            patch(
                "framework.inference.flux2_klein.utils.harmonize_with_mask",
                side_effect=lambda _p, image, _m: image,
            ),
            patch(
                "framework.inference.flux2_klein.utils.vitmatte_alpha",
                side_effect=lambda _pool, _img, m: m.astype(float),
            ),
        ):
            result = flux2_klein_edit_object(
                self.pool,
                self.image,
                self.mask,
                "x",
                **kwargs,
            )
        return result, crops

    def test_paste_weight_is_the_matte(self):
        result, _ = self._edit(
            lambda crop: Image.new("RGB", crop.size, (30, 30, 200)), mode="local"
        )
        self.assertEqual(result.getpixel((120, 120)), (30, 30, 200))
        self.assertEqual(result.getpixel((200, 120)), (100, 100, 100))

    def test_pixels_outside_matte_are_untouched(self):
        def paint(crop):
            out = Image.new("RGB", crop.size, (255, 0, 0))
            return out

        result, _ = self._edit(paint, mode="regional", padding=0.5)
        self.assertEqual(result.getpixel((150, 120)), (100, 100, 100))
        self.assertEqual(result.getpixel((120, 120)), (255, 0, 0))

    def test_global_returns_full_klein_output(self):
        result, crops = self._edit(
            lambda crop: Image.new("RGB", crop.size, "red"), mode="global"
        )
        self.assertEqual(crops, [(240, 240)])
        self.assertEqual(result.getpixel((5, 5)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
