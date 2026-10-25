import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from framework.inference.flux2_klein import replace_object as module
from framework.inference.flux2_klein.replace_object import (
    _old_area_weight,
    flux2_klein_replace_object,
)


class ReplaceTests(unittest.TestCase):
    def test_old_area_weight_covers_the_object(self):
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (40, 40, 60, 60))
        weight = _old_area_weight(mask, (20, 20, 80, 80))
        self.assertEqual(weight.shape, (60, 60))
        self.assertGreater(weight[30, 30], 0.99)
        self.assertEqual(weight[0, 0], 0.0)

    def test_replace_klein_runs_once_and_old_area_is_pasted(self):
        image = Image.new("RGB", (100, 100), (10, 10, 10))
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (40, 40, 60, 60))
        pool = SimpleNamespace(
            config=SimpleNamespace(klein=SimpleNamespace(mask_context_padding=20)),
            flux2_klein_pipeline=object(),
        )
        klein = Image.new("RGB", (60, 60), (200, 200, 200))
        new_object = np.zeros((60, 60), bool)
        new_object[25:35, 25:35] = True
        calls = []

        def fake_generate(pipeline, working, prompt, config, seed_offset=0):
            calls.append(prompt)
            return klein.resize(working.size)

        with (
            patch.object(module, "generate_klein_edit", fake_generate),
            patch.object(
                module,
                "segment_added_object",
                lambda pool, before, after, region, prompt, debug=None: new_object,
            ),
            patch.object(
                module,
                "blend_edit",
                lambda pool, scene, crop_box, after, obj, **kwargs: (
                    scene,
                    kwargs["extra_weight"],
                ),
            ),
        ):
            result = flux2_klein_replace_object(pool, image, mask, "a dog")
        self.assertEqual(result.size, image.size)
        self.assertEqual(len(calls), 1)
        self.assertIn("a dog", calls[0])


if __name__ == "__main__":
    unittest.main()
