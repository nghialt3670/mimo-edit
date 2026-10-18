import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from framework.inference.config import KleinInferenceConfig
from framework.inference.flux2_klein import flux2_klein_edit_object
from framework.inference.flux2_klein.utils import matte_alpha


class MatteAlphaTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((60, 60), bool)
        self.mask[20:40, 20:40] = True
        self.limit = np.ones((60, 60), bool)

    def test_fractional_values_are_preserved(self):
        matte = np.zeros((60, 60), np.float32)
        matte[20:40, 20:40] = 1.0
        matte[30, 40:44] = 0.37  # fur strand outside the mask
        alpha = matte_alpha(matte, None, self.mask, self.limit)
        self.assertAlmostEqual(float(alpha[30, 42]), 0.37, places=5)

    def test_matte_objects_not_touching_the_mask_are_dropped(self):
        matte = np.zeros((60, 60), np.float32)
        matte[20:40, 20:40] = 1.0
        matte[2:8, 2:8] = 1.0  # another object
        alpha = matte_alpha(matte, None, self.mask, self.limit)
        self.assertEqual(float(alpha[4, 4]), 0.0)
        self.assertEqual(float(alpha[30, 30]), 1.0)

    def test_leakage_refinement_reverts_unchanged_background_only(self):
        matte = np.zeros((60, 60), np.float32)
        matte[20:40, 20:40] = 1.0
        matte[30, 40:44] = 1.0  # strand Klein really changed
        matte[10:20, 25:35] = 1.0  # background leakage Klein did not change
        residual = np.zeros((60, 60), np.float32)
        residual[30, 40:44] = 60.0
        alpha = matte_alpha(matte, residual, self.mask, self.limit)
        self.assertEqual(float(alpha[30, 42]), 1.0)
        self.assertEqual(float(alpha[15, 30]), 0.0)
        self.assertEqual(float(alpha[30, 30]), 1.0)


class EditObjectMattingTests(unittest.TestCase):
    def test_matte_alpha_drives_the_paste(self):
        image = Image.new("RGB", (240, 240), (100, 100, 100))
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (100, 100, 140, 140))
        pool = SimpleNamespace(
            flux2_klein_pipeline=object(),
            config=SimpleNamespace(klein=KleinInferenceConfig()),
            pctnet_pipeline=object(),
        )
        seen = []

        def fake_klein(_pipeline, crop, _prompt, _config):
            return Image.new("RGB", crop.size, (30, 30, 200))

        def fake_matte(_pool, edited):
            seen.append(edited.size)
            matte = np.zeros((edited.height, edited.width), np.float32)
            matte[30:90, 30:90] = 0.5
            return matte

        with (
            patch(
                "framework.inference.flux2_klein.edit_object.generate_klein_edit",
                fake_klein,
            ),
            patch(
                "framework.inference.flux2_klein.edit_object.birefnet_alpha_matte",
                fake_matte,
            ),
        ):
            result = flux2_klein_edit_object(
                pool, image, mask, "x", harmonize=False, matting=True
            )
        self.assertEqual(len(seen), 1)
        self.assertEqual(result.getpixel((120, 120)), (65, 65, 150))
        self.assertEqual(result.getpixel((5, 5)), (100, 100, 100))


if __name__ == "__main__":
    unittest.main()
