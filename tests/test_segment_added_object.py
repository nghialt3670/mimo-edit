import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from framework.inference.flux2_klein.utils import segment_added_object


def _scene():
    before = Image.new("RGB", (100, 100), (90, 90, 90))
    after = before.copy()
    after.paste((200, 30, 30), (40, 40, 60, 60))  # the added object
    region = np.zeros((100, 100), bool)
    region[20:80, 20:80] = True  # the box is much larger than the object
    return before, after, region


def _sam(mask_box):
    mask = np.zeros((100, 100), np.uint8)
    x1, y1, x2, y2 = mask_box
    mask[y1:y2, x1:x2] = 255
    return patch(
        "framework.inference.sam3.sam3_segment_mask", return_value=Image.fromarray(mask)
    )


class SegmentAddedObjectTests(unittest.TestCase):
    def test_mask_follows_the_object_not_the_box(self):
        before, after, region = _scene()
        with _sam((40, 40, 60, 60)) as sam:
            mask = segment_added_object(SimpleNamespace(), before, after, region)
        self.assertEqual(int(mask.sum()), 400)
        self.assertEqual(sam.call_args.kwargs["box"], [20.0, 20.0, 80.0, 80.0])

    def test_mask_is_limited_to_the_region(self):
        before, after, region = _scene()
        with _sam((0, 0, 100, 100)):
            mask = segment_added_object(SimpleNamespace(), before, after, region)
        self.assertFalse(mask[~region].any())

    def test_unchanged_segment_falls_back_to_the_region(self):
        before, after, region = _scene()
        with _sam((22, 22, 38, 38)):  # SAM picked old content, not the new object
            mask = segment_added_object(SimpleNamespace(), before, after, region)
        self.assertTrue(np.array_equal(mask, region))

    def test_tiny_segment_falls_back_to_the_region(self):
        before, after, region = _scene()
        with _sam((50, 50, 52, 52)):
            mask = segment_added_object(SimpleNamespace(), before, after, region)
        self.assertTrue(np.array_equal(mask, region))


if __name__ == "__main__":
    unittest.main()
