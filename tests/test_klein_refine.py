import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from framework.inference.flux2_klein.refine import klein_refine, refine_zone


class _Pipe:
    def __init__(self, color):
        self.color = color
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), self.color)
        return SimpleNamespace(images=[image])


def _pool(pipe):
    klein = SimpleNamespace(num_inference_steps=1, guidance_scale=1.0)
    return SimpleNamespace(
        flux2_klein_pipeline=pipe, config=SimpleNamespace(klein=klein)
    )


class KleinRefineTests(unittest.TestCase):
    def test_zone_is_full_on_the_object_and_zero_far_away(self):
        matte = np.zeros((300, 300), np.float32)
        matte[100:140, 100:140] = 1
        zone = refine_zone(matte)
        self.assertEqual(zone[120, 120], 1.0)
        self.assertEqual(zone[0, 0], 0.0)
        self.assertGreater(zone[120, 150], 0.9)

    def test_background_far_from_the_object_stays_untouched(self):
        composite = Image.new("RGB", (320, 240), (10, 20, 30))
        matte = np.zeros((240, 320), np.float32)
        matte[100:140, 150:190] = 1
        pipe = _Pipe((200, 0, 0))
        result, raw = klein_refine(_pool(pipe), composite, matte)
        self.assertEqual(result.size, composite.size)
        self.assertEqual(raw.size, composite.size)
        self.assertEqual(result.getpixel((160, 120)), (200, 0, 0))
        self.assertEqual(result.getpixel((2, 2)), (10, 20, 30))
        self.assertEqual(pipe.calls[0]["width"] % 16, 0)
        self.assertEqual(pipe.calls[0]["height"] % 16, 0)


if __name__ == "__main__":
    unittest.main()
