import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from framework.inference.flux2_klein import shadow


def _scene():
    image = np.full((200, 200, 3), 180, np.uint8)
    image[100:140, 60:120] = 60  # shadow beside the object
    image[40:100, 60:120] = (200, 40, 40)  # object
    obj = np.zeros((200, 200), np.uint8)
    obj[40:100, 60:120] = 255
    return Image.fromarray(image), obj


class FindShadowMaskTest(unittest.TestCase):
    def test_heuristic_finds_dark_region_touching_object(self):
        image, obj = _scene()
        with patch.object(shadow, "sam3_segment_masks", side_effect=RuntimeError):
            result = shadow.find_shadow_mask(None, image, obj)
        self.assertTrue(result[120, 90] > 0)
        self.assertEqual(result[60, 90], 0)
        self.assertEqual(result[10, 10], 0)

    def test_sam3_segment_touching_object_is_used(self):
        image, obj = _scene()
        sam = np.zeros((200, 200), np.uint8)
        sam[100:140, 60:120] = 255
        far = np.zeros((200, 200), np.uint8)
        far[0:20, 150:190] = 255
        with patch.object(
            shadow,
            "sam3_segment_masks",
            return_value=[Image.fromarray(far), Image.fromarray(sam)],
        ):
            result = shadow.find_shadow_mask(None, image, obj)
        self.assertTrue(result[120, 90] > 0)
        self.assertEqual(result[10, 170], 0)

    def test_large_scene_shadow_touching_object_is_rejected(self):
        image, obj = _scene()
        scene = np.zeros((200, 200), np.uint8)
        scene[90:200, 0:200] = 255  # huge shadow that merely touches the object
        with patch.object(
            shadow, "sam3_segment_masks", return_value=[Image.fromarray(scene)]
        ):
            plain = Image.fromarray(np.full((200, 200, 3), 150, np.uint8))
            result = shadow.find_shadow_mask(None, plain, obj)
        self.assertEqual(result.max(), 0)

    def test_oversized_shadow_is_ignored(self):
        image, obj = _scene()
        huge = np.full((200, 200), 255, np.uint8)
        with patch.object(
            shadow, "sam3_segment_masks", return_value=[Image.fromarray(huge)]
        ):
            plain = Image.fromarray(np.full((200, 200, 3), 150, np.uint8))
            result = shadow.find_shadow_mask(None, plain, obj)
        self.assertEqual(result.max(), 0)


if __name__ == "__main__":
    unittest.main()
