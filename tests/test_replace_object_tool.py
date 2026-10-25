from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from framework.tools.image_editing.replace_object import replace_object


class ReplaceObjectToolTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (20, 16), "white")
        self.mask = Image.new("L", self.image.size, 0)
        self.mask.paste(255, (4, 3, 12, 11))
        self.ctx = SimpleNamespace(
            model_pool=SimpleNamespace(
                flux2_klein_pipeline=object(),
                config=SimpleNamespace(klein=object()),
                pctnet_pipeline=object(),
            )
        )

    def test_prompt_replacement_calls_direct_replace(self):
        with (
            patch(
                "framework.tools.image_editing.replace_object.flux2_klein_remove_object"
            ) as remove,
            patch(
                "framework.tools.image_editing.replace_object.flux2_klein_replace_object",
                return_value=self.image,
            ) as replace,
        ):
            result = replace_object(
                self.ctx,
                self.image,
                self.mask,
                prompt="a small red boat",
                anchor="on the water",
            )

        self.assertIs(result, self.image)
        remove.assert_not_called()
        replace.assert_called_once_with(
            self.ctx.model_pool,
            self.image,
            self.mask,
            "a small red boat",
            anchor="on the water",
        )

    def test_image_replacement_keeps_existing_remove_and_paste_behavior(self):
        cleaned = Image.new("RGB", self.image.size, "gray")
        replacement = Image.new("RGB", (3, 3), "red")
        with (
            patch(
                "framework.tools.image_editing.replace_object.flux2_klein_remove_object",
                return_value=cleaned,
            ) as remove,
            patch(
                "framework.tools.image_editing.replace_object.flux2_klein_replace_object"
            ) as add,
        ):
            result = replace_object(
                self.ctx,
                self.image,
                self.mask,
                replacement=replacement,
            )

        remove.assert_called_once_with(self.ctx.model_pool, self.image, self.mask)
        add.assert_not_called()
        self.assertEqual(result.getpixel((5, 4)), (255, 0, 0))
        self.assertEqual(result.getpixel((0, 0)), (128, 128, 128))

    def test_requires_exactly_one_replacement_input(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            replace_object(self.ctx, self.image, self.mask)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            replace_object(
                self.ctx,
                self.image,
                self.mask,
                replacement=Image.new("RGB", (2, 2)),
                prompt="a replacement",
            )


if __name__ == "__main__":
    unittest.main()
