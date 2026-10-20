import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from framework.inference.flux2_klein.placement import describe_anchor, placement
from framework.inference.flux2_klein.shadow_transfer import transfer_shadow


def _scene(shadow: bool, drift: float = 1.0):
    before = np.full((120, 120, 3), 160, np.uint8)
    after = before.astype(np.float32) * drift
    matte = np.zeros((120, 120), np.float32)
    matte[30:70, 40:70] = 1.0
    after[30:70, 40:70] = (200, 60, 60)
    if shadow:
        after[70:80, 40:80] *= 0.6  # darker, same hue, touching the object
        after[100:110, 5:15] *= 0.6  # far away: not its shadow
    return (
        Image.fromarray(before),
        Image.fromarray(after.clip(0, 255).astype(np.uint8)),
        matte,
    )


class TransferShadowTests(unittest.TestCase):
    def test_transfers_the_shadow_touching_the_object_only(self):
        before, after, matte = _scene(shadow=True)
        shade, source = transfer_shadow(before, after, matte)
        self.assertEqual(source, "transfer")
        self.assertLess(shade[75, 55], 0.75)
        self.assertGreater(shade[105, 10], 0.99)
        self.assertTrue(np.all(shade[matte > 0.5] > 0.99))

    def test_global_exposure_drift_is_not_a_shadow(self):
        before, after, matte = _scene(shadow=False, drift=0.8)
        shade, source = transfer_shadow(before, after, matte)
        self.assertEqual(source, "none")
        self.assertTrue(np.all(shade == 1))

    def test_ground_shadow_is_synthesised_only_when_asked(self):
        before, after, matte = _scene(shadow=False)
        self.assertEqual(transfer_shadow(before, after, matte)[1], "none")
        shade, source = transfer_shadow(before, after, matte, ground=True)
        self.assertEqual(source, "synthetic")
        self.assertLess(shade[72, 55], 0.9)
        self.assertGreaterEqual(float(shade.min()), 0.6)


class PlacementTests(unittest.TestCase):
    def test_placement_phrase(self):
        self.assertEqual(placement("on the blue cooler."), " on the blue cooler")
        self.assertEqual(placement(None), "")

    def test_describe_anchor_uses_the_model_text(self):
        seen = {}

        def generate_content(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(text=" On the blue cooler. ")

        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        image = Image.new("RGB", (100, 100), "gray")
        self.assertEqual(
            describe_anchor(image, (10, 10, 50, 50), "a sticker", client=client),
            "On the blue cooler",
        )
        self.assertIn("a sticker", seen["contents"][0])

    def test_describe_anchor_returns_none_on_failure_or_rambling(self):
        def boom(**_):
            raise RuntimeError("no network")

        image = Image.new("RGB", (100, 100), "gray")
        for generate in (boom, lambda **_: SimpleNamespace(text="word " * 30)):
            client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
            self.assertIsNone(
                describe_anchor(image, (0, 0, 10, 10), "a dog", client=client)
            )


if __name__ == "__main__":
    unittest.main()
