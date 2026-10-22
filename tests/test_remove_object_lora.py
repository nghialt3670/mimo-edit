import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from framework.inference.config import KleinInferenceConfig
from framework.inference.flux2_klein.lora import (
    OBJECT_REMOVE_ADAPTER_NAME,
    SLOPSHOP_ERASE_ADAPTER_NAME,
    SLOPSHOP_ERASE_PROMPT,
)
from framework.inference.flux2_klein.remove_object import flux2_klein_remove_object


class RemoveObjectLoraTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 80), (10, 20, 30))
        self.mask = Image.new("L", self.image.size, 0)
        self.mask.paste(255, (30, 30, 50, 50))
        self.pool = SimpleNamespace(
            flux2_klein_pipeline=object(),
            config=SimpleNamespace(klein=KleinInferenceConfig()),
        )
        self.captured = {}

    def _generate(self, _pipe, image, prompt, _config, **kwargs):
        self.captured["image"] = image
        self.captured["prompt"] = prompt
        self.captured.update(kwargs)
        size = image[0].size if isinstance(image, list) else image.size
        return Image.new("RGB", size, (0, 255, 0))

    def _remove(self, **kwargs):
        with (
            patch(
                "framework.inference.flux2_klein.remove_object.generate_klein",
                self._generate,
            ),
            patch(
                "framework.inference.flux2_klein.remove_object.validated_object_remove_lora_path",
                return_value=Path("/tmp/remove.safetensors"),
            ),
            patch(
                "framework.inference.flux2_klein.remove_object.load_remove_prompt_embeds",
                return_value=None,
            ),
        ):
            return flux2_klein_remove_object(
                self.pool, self.image, self.mask, feather_px=0, **kwargs
            )

    def test_fal_highlights_a_single_image(self):
        self._remove(lora="fal")
        image = self.captured["image"]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(self.captured["adapter_name"], OBJECT_REMOVE_ADAPTER_NAME)
        self.assertEqual(self.captured["adapter_weight"], 1.3)
        r, g, b = image.getpixel((image.width // 2, image.height // 2))
        self.assertGreater(r, g)
        self.assertGreater(b, g)

    def test_slopshop_passes_photo_and_rgb_mask(self):
        self._remove(lora="slopshop")
        image = self.captured["image"]
        self.assertIsInstance(image, list)
        self.assertEqual(len(image), 2)
        photo, mask = image
        self.assertEqual(self.captured["adapter_name"], SLOPSHOP_ERASE_ADAPTER_NAME)
        self.assertEqual(self.captured["adapter_weight"], 1.0)
        self.assertEqual(self.captured["prompt"], SLOPSHOP_ERASE_PROMPT)
        self.assertEqual(
            photo.getpixel((photo.width // 2, photo.height // 2)), (10, 20, 30)
        )
        self.assertEqual(mask.mode, "RGB")
        self.assertEqual(
            mask.getpixel((mask.width // 2, mask.height // 2)), (255, 255, 255)
        )

    def test_config_selects_slopshop(self):
        self.pool.config = SimpleNamespace(
            klein=KleinInferenceConfig(remove_lora="slopshop")
        )
        self._remove()
        self.assertEqual(self.captured["adapter_name"], SLOPSHOP_ERASE_ADAPTER_NAME)


if __name__ == "__main__":
    unittest.main()
