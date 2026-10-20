import unittest

import numpy as np
from PIL import Image

from framework.inference.flux2_klein.remove_object import _poisson_blend


class PoissonEdgeTests(unittest.TestCase):
    def _scene(self):
        yy, xx = np.mgrid[0:120, 0:160]
        clean = np.stack([xx * 1.5, yy * 2, xx + yy], -1).clip(0, 255).astype(np.uint8)
        reference = clean.copy()
        reference[40:80, 110:160] = (200, 30, 30)  # object touching the right edge
        mask = np.zeros((120, 160), np.uint8)
        mask[40:80, 110:160] = 255
        return Image.fromarray(clean), Image.fromarray(reference), Image.fromarray(mask)

    def test_object_touching_the_image_edge_is_replaced_up_to_the_edge(self):
        fill, reference, mask = self._scene()
        out = np.asarray(_poisson_blend(fill, reference, mask, 2)).astype(int)
        original = np.asarray(reference).astype(int)
        for column in range(159, 155, -1):
            unchanged = np.abs(out[40:80, column] - original[40:80, column]).max(1) <= 6
            self.assertLess(float(unchanged.mean()), 0.1, f"column {column}")

    def test_pixels_outside_the_mask_are_untouched(self):
        fill, reference, mask = self._scene()
        out = np.asarray(_poisson_blend(fill, reference, mask, 0))
        outside = np.asarray(mask) == 0
        self.assertTrue(np.array_equal(out[outside], np.asarray(reference)[outside]))


if __name__ == "__main__":
    unittest.main()
