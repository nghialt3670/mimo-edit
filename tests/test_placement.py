import unittest
from types import SimpleNamespace

from PIL import Image

from framework.inference.flux2_klein.placement import describe_anchor, placement


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

