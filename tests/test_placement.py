import unittest

from framework.inference.flux2_klein.placement import placement


class PlacementTests(unittest.TestCase):
    def test_placement_phrase(self):
        self.assertEqual(placement("on the blue cooler."), " on the blue cooler")
        self.assertEqual(placement(None), "")


if __name__ == "__main__":
    unittest.main()

