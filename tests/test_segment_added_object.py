import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from framework.inference.flux2_klein.utils import NothingAdded, segment_added_object


def _scene():
    before = Image.new("RGB", (100, 100), (90, 90, 90))
    after = before.copy()
    after.paste((200, 30, 30), (40, 40, 60, 60))  # the added object
    region = np.zeros((100, 100), bool)
    region[20:80, 20:80] = True  # the box is much larger than the object
    return before, after, region


def _sam(*boxes):
    masks = []
    for x1, y1, x2, y2 in boxes:
        mask = np.zeros((100, 100), np.uint8)
        mask[y1:y2, x1:x2] = 255
        masks.append(Image.fromarray(mask))
    return patch("framework.inference.sam3.sam3_segment_masks", return_value=masks)


class SegmentAddedObjectTests(unittest.TestCase):
    def run_segment(self, before, after, region, prompt="a thing", **kwargs):
        return segment_added_object(
            SimpleNamespace(), before, after, region, prompt, **kwargs
        )

    def test_text_mask_follows_the_object_not_the_box(self):
        before, after, region = _scene()
        debug = {}
        with _sam((38, 38, 62, 62)) as sam:
            mask = self.run_segment(before, after, region, debug=debug)
        self.assertEqual(int(mask.sum()), 24 * 24)
        self.assertEqual(debug["mask_source"], "text")
        self.assertEqual(sam.call_args.args[2], "a thing")

    def test_old_content_candidate_is_ignored_for_the_change_map(self):
        before, after, region = _scene()
        debug = {}
        with _sam((22, 22, 38, 38)):  # existing content, not the new object
            mask = self.run_segment(before, after, region, debug=debug)
        self.assertEqual(debug["mask_source"], "change")
        self.assertTrue(mask[50, 50])
        self.assertFalse(mask[25, 25])

    def test_candidate_centred_outside_the_region_is_ignored(self):
        before, after, region = _scene()
        after.paste((30, 200, 30), (0, 0, 10, 10))
        with _sam((0, 0, 10, 10)):
            mask = self.run_segment(before, after, region)
        self.assertFalse(mask[:10, :10].any())

    def test_object_may_extend_past_the_region(self):
        before = Image.new("RGB", (100, 100), (90, 90, 90))
        after = before.copy()
        after.paste((200, 30, 30), (30, 30, 90, 70))
        region = np.zeros((100, 100), bool)
        region[30:70, 30:60] = True
        with _sam((30, 30, 90, 70)):
            mask = self.run_segment(before, after, region)
        self.assertEqual(int(mask.sum()), 60 * 40)

    def test_unchanged_output_raises(self):
        before, _, region = _scene()
        with self.assertRaises(NothingAdded):
            self.run_segment(before, before.copy(), region)

    def test_never_returns_the_whole_region(self):
        before, after, region = _scene()
        with _sam():
            mask = self.run_segment(before, after, region)
        self.assertFalse(np.array_equal(mask, region))


if __name__ == "__main__":
    unittest.main()
