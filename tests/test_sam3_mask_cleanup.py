import unittest

import numpy as np

from framework.inference.sam3 import _keep_prompt_components


class KeepPromptComponentsTests(unittest.TestCase):
    def test_drops_isolated_speckles_and_keeps_prompt_component(self):
        binary = np.zeros((20, 20), dtype="uint8")
        binary[5:10, 5:10] = 1
        binary[18, 18] = 1

        result = _keep_prompt_components(binary, [{"x": 6, "y": 6, "label": 1}])

        self.assertEqual(result.sum(), 25)
        self.assertEqual(result[18, 18], 0)

    def test_falls_back_to_largest_component_for_box_prompts(self):
        binary = np.zeros((20, 20), dtype="uint8")
        binary[2:4, 2:4] = 1
        binary[10:16, 10:16] = 1

        result = _keep_prompt_components(binary, None)

        self.assertEqual(result.sum(), 36)
        self.assertEqual(result[2, 2], 0)

    def test_empty_mask_is_returned_unchanged(self):
        binary = np.zeros((4, 4), dtype="uint8")

        result = _keep_prompt_components(binary, [{"x": 1, "y": 1, "label": 1}])

        self.assertEqual(result.sum(), 0)


if __name__ == "__main__":
    unittest.main()
