import unittest

import numpy as np
from PIL import Image

from framework.inference.flux2_klein.utils import change_alpha


class ChangeAlphaThinStructureTest(unittest.TestCase):
    def setUp(self):
        before = np.full((80, 80, 3), 25, np.uint8)
        after = before.copy()
        after[20:60, 20:60] = (20, 60, 220)  # recolored body
        after[38:41, 60:70] = (20, 60, 220)  # thin wisp outside the mask
        after[5, 5] = (200, 0, 0)  # isolated speck far away
        self.before, self.after = Image.fromarray(before), Image.fromarray(after)
        self.mask = np.zeros((80, 80), bool)
        self.mask[22:58, 22:58] = True
        self.limit = np.ones((80, 80), bool)

    def test_wisp_outside_mask_follows_klein_not_the_mask(self):
        alpha = change_alpha(self.before, self.after, self.mask, self.limit)
        self.assertGreater(alpha[39, 65], 0.9)
        self.assertEqual(alpha[21, 21] > 0.9, True)
        self.assertLess(alpha[30, 72], 0.1)  # unchanged pixels stay original
        self.assertLess(alpha[5, 5], 0.1)

    def test_global_tone_drift_is_not_pasted(self):
        drifted = np.asarray(self.after).astype(int) + 12
        after = Image.fromarray(np.clip(drifted, 0, 255).astype(np.uint8))
        alpha = change_alpha(self.before, after, self.mask, self.limit, reach=6)
        self.assertLess(alpha[70, 10], 0.1)
        self.assertGreater(alpha[40, 40], 0.9)


if __name__ == "__main__":
    unittest.main()
