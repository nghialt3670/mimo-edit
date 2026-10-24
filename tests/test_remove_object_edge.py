from __future__ import annotations

import unittest

import numpy as np
from PIL import Image


class RemoveObjectEdgeTests(unittest.TestCase):
    def test_edge_pad_is_zero_when_mask_is_interior(self):
        from framework.inference.flux2_klein.remove_object import _edge_pad

        mask = np.zeros((100, 100), np.uint8)
        mask[40:60, 40:60] = 255

        self.assertEqual(_edge_pad(Image.fromarray(mask, "L"), 100), 0)

    def test_edge_pad_is_positive_when_mask_touches_border(self):
        from framework.inference.flux2_klein.remove_object import EDGE_PAD, _edge_pad

        mask = np.zeros((300, 300), np.uint8)
        mask[120:180, 0:20] = 255

        self.assertEqual(_edge_pad(Image.fromarray(mask, "L"), 300), EDGE_PAD)

    def test_reflect_pad_keeps_mask_inside_padded_frame(self):
        from framework.inference.flux2_klein.remove_object import _reflect_pad

        image = Image.new("RGB", (50, 40), (10, 20, 30))
        mask = Image.new("L", (50, 40), 0)
        mask.paste(255, (0, 10, 10, 20))

        padded_image, padded_mask = _reflect_pad(image, mask, 8)

        self.assertEqual(padded_image.size, (66, 56))
        self.assertEqual(padded_mask.size, (66, 56))
        self.assertTrue(np.asarray(padded_mask)[:, 0].any())

    def test_strip_highlight_replaces_leaked_magenta_inside_mask(self):
        from framework.inference.flux2_klein.remove_object import _strip_highlight

        result = Image.new("RGB", (4, 4), (255, 0, 255))
        fill = Image.new("RGB", (4, 4), (90, 100, 110))
        mask = Image.new("L", (4, 4), 0)
        mask.paste(255, (0, 0, 2, 4))

        out = np.asarray(_strip_highlight(result, fill, mask))

        self.assertEqual(tuple(out[0, 0]), (90, 100, 110))
        self.assertEqual(tuple(out[0, 3]), (255, 0, 255))


if __name__ == "__main__":
    unittest.main()
