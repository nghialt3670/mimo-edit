import unittest

import numpy as np
from PIL import Image

from framework.inference.object_clear import _pad_mask


class PadMaskTest(unittest.TestCase):
    def test_dilates_by_ratio_of_bbox_longer_side(self):
        mask = Image.new("L", (200, 200), 0)
        mask.paste(255, (80, 90, 120, 110))  # 40x20 -> radius 2
        padded = _pad_mask(mask, 0.05)
        self.assertEqual(padded.getbbox(), (78, 88, 122, 112))
        self.assertTrue(np.asarray(padded)[mask.getbbox()[1] : mask.getbbox()[3]].any())

    def test_zero_ratio_or_empty_mask_is_unchanged(self):
        mask = Image.new("L", (50, 50), 0)
        self.assertIsNone(_pad_mask(mask, 0.05).getbbox())
        mask.paste(255, (10, 10, 20, 20))
        self.assertEqual(_pad_mask(mask, 0).getbbox(), (10, 10, 20, 20))


if __name__ == "__main__":
    unittest.main()
