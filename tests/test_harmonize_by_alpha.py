import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from framework.inference.flux2_klein.utils import blend_edit, harmonize_by_alpha


def _brighten(_predictor, image, mask):
    # Stand-in for PCT-Net: +100 everywhere (binary mask is only inspected).
    assert set(np.unique(np.asarray(mask))) <= {0, 255}
    return Image.fromarray(
        np.clip(np.asarray(image, int) + 100, 0, 255).astype("uint8")
    )


class HarmonizeByAlphaTests(unittest.TestCase):
    def test_strength_scales_with_alpha(self):
        image = Image.new("RGB", (4, 1), (50, 50, 50))
        alpha = np.array([[0.0, 0.25, 0.5, 1.0]], np.float32)
        with patch(
            "framework.inference.flux2_klein.utils.harmonize_with_mask", _brighten
        ):
            out = np.asarray(harmonize_by_alpha(object(), image, alpha))
        self.assertEqual(out[0, :, 0].tolist(), [50, 75, 100, 150])

    def test_blend_edit_harmonizes_the_whole_scene_where_matte_is_not_black(self):
        scene = Image.new("RGB", (8, 1), (0, 0, 0))
        crop_box = (2, 0, 6, 1)
        after = Image.new("RGB", (4, 1), (50, 50, 50))
        matte = np.array([[0.0, 0.5, 1.0, 1.0]], np.float32)
        pool = SimpleNamespace(pctnet_pipeline=object())
        seen = []

        def spy(predictor, image, mask):
            seen.append((image.size, mask.getbbox()))
            return _brighten(predictor, image, mask)

        with (
            patch(
                "framework.inference.flux2_klein.utils.vitmatte_alpha",
                return_value=matte,
            ),
            patch("framework.inference.flux2_klein.utils.harmonize_with_mask", spy),
            patch(
                "framework.inference.flux2_klein.utils.semantic_diff_weight",
                return_value=(matte, matte),
            ),
        ):
            result, _ = blend_edit(pool, scene, crop_box, after, matte > 0.5)
            plain, _ = blend_edit(
                pool, scene, crop_box, after, matte > 0.5, harmonize=False
            )
        # PCT-Net got the full 8x1 scene and a mask limited to matte > 0.
        self.assertEqual(seen, [((8, 1), (3, 0, 6, 1))])
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((2, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((5, 0)), (150, 150, 150))
        self.assertEqual(plain.getpixel((5, 0)), (50, 50, 50))
        self.assertEqual(plain.getpixel((7, 0)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
