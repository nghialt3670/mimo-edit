from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

        with patch(
            "framework.inference.flux2_klein.add_object.generate_klein_edit",
            return_value=edited,
        ):
            result = flux2_klein_add_object_with_mask(
                _pool(mask_context_padding=10),
                image,
                mask,
                "a red object",
                harmonize=False,
            )

        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((60, 50)), (255, 0, 0))
        self.assertEqual(result.getpixel((45, 50)), (0, 0, 0))
        self.assertEqual(result.getpixel((100, 100)), (0, 0, 0))

    def test_mask_add_harmonizes_the_diff_region(self):
        image = Image.new("RGB", (200, 160), "black")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (50, 40, 80, 70))
        pool = _pool(mask_context_padding=10)

        with (
            patch(
                "framework.inference.flux2_klein.add_object.generate_klein_edit",
                return_value=_red_square(),
            ),
            patch(
                "framework.inference.external.pctnet_harmonization.harmonize_pctnet",
                side_effect=lambda _predictor, img, region: img,
            ) as harmonize,
        ):
            flux2_klein_add_object_with_mask(pool, image, mask, "a red object")

        harmonize.assert_called_once()
        self.assertIs(harmonize.call_args.args[0], pool.pctnet_pipeline)
        # Harmonization runs in crop coordinates (crop origin is (40, 30)).
        left, top, right, bottom = harmonize.call_args.args[2].getbbox()
        self.assertLessEqual((left, top), (10, 10))
        self.assertGreaterEqual((right, bottom), (40, 40))

    def test_box_add_harmonizes_diff_region_in_box(self):
        from framework.inference.flux2_klein import flux2_klein_add_object_with_box

        image = Image.new("RGB", (64, 64), "black")
        pipe = lambda **kwargs: type(
            "Out",
            (),
            {"images": [Image.new("RGB", (kwargs["width"], kwargs["height"]), "red")]},
        )()
        with patch(
            "framework.inference.external.pctnet_harmonization.harmonize_pctnet",
            side_effect=lambda _predictor, img, region: img,
        ) as harmonize:
            flux2_klein_add_object_with_box(_pool(pipe), image, (16, 16, 48, 48), "x")

        harmonize.assert_called_once()
        mask = harmonize.call_args.args[2]
        self.assertIsNotNone(mask.getbbox())
        left, top, right, bottom = mask.getbbox()
        # Crop coordinates: the crop is the 32x32 box.
        self.assertTrue(left >= 0 and top >= 0 and right <= 32 and bottom <= 32)

    def test_harmonize_false_skips_pctnet(self):
        image = Image.new("RGB", (64, 64), "black")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (16, 16, 32, 32))
        with (
            patch(
                "framework.inference.flux2_klein.add_object.generate_klein_edit",
                return_value=Image.new("RGB", (64, 64), "red"),
            ),
            patch(
                "framework.inference.external.pctnet_harmonization.harmonize_pctnet"
            ) as harmonize,
        ):
            flux2_klein_add_object_with_mask(_pool(), image, mask, "x", harmonize=False)
        harmonize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
