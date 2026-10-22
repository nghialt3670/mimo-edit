import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from framework.inference.config import KleinInferenceConfig
from framework.inference.flux2_klein.lora import (
    CONSISTENCY_ADAPTER_NAME,
    OBJECT_REMOVE_ADAPTER_NAME,
    SLOPSHOP_ERASE_ADAPTER_NAME,
    generate_klein,
    remove_lora_spec,
)
from framework.inference.flux2_klein.utils import generate_klein_edit


class FakePipeline:
    def __init__(self):
        self.loaded: dict[str, tuple[str, str]] = {}
        self.active: list[str] = []
        self.weights = None
        self.calls: list[tuple] = []
        self.unload_count = 0
        self.delete_count = 0

    def get_list_adapters(self):
        return {"transformer": list(self.loaded)}

    def get_active_adapters(self):
        return list(self.active)

    def load_lora_weights(self, pretrained, weight_name=None, adapter_name=None):
        self.loaded[adapter_name] = (pretrained, weight_name)
        self.active = [adapter_name]
        self.calls.append(("load", adapter_name, pretrained, weight_name))

    def set_adapters(self, names, adapter_weights=None):
        if isinstance(names, str):
            names = [names]
        self.active = list(names)
        self.weights = adapter_weights
        self.calls.append(("set", list(names), adapter_weights))

    def enable_lora(self):
        self.calls.append(("enable",))

    def disable_lora(self):
        self.active = []
        self.calls.append(("disable",))

    def unload_lora_weights(self):
        self.unload_count += 1
        self.loaded.clear()
        self.active = []

    def delete_adapters(self, _name):
        self.delete_count += 1

    def __call__(self, **kwargs):
        self.calls.append(("infer", tuple(self.active), self.weights))
        self.calls.append(("infer_kwargs", kwargs))
        size = (kwargs.get("width") or 8, kwargs.get("height") or 8)
        return SimpleNamespace(images=[Image.new("RGB", size, "red")])


class KleinLoraTests(unittest.TestCase):
    def setUp(self):
        self.pipe = FakePipeline()
        self.image = Image.new("RGB", (8, 8), "blue")
        self.config = KleinInferenceConfig()

    def test_two_adapters_stay_loaded_and_switch_exclusively(self):
        generate_klein_edit(
            self.pipe, self.image, "edit", self.config, lora_strength=1.0
        )
        generate_klein(
            self.pipe,
            self.image,
            "remove",
            self.config,
            adapter_name=OBJECT_REMOVE_ADAPTER_NAME,
            adapter_weight=1.3,
            pretrained="/weights",
            weight_name="remove.safetensors",
        )

        self.assertEqual(
            set(self.pipe.loaded),
            {CONSISTENCY_ADAPTER_NAME, OBJECT_REMOVE_ADAPTER_NAME},
        )
        self.assertEqual(self.pipe.unload_count, 0)
        self.assertEqual(self.pipe.delete_count, 0)
        self.assertEqual(self.pipe.active, [])

        sets = [call for call in self.pipe.calls if call[0] == "set"]
        self.assertEqual(
            sets,
            [
                ("set", [CONSISTENCY_ADAPTER_NAME], [1.0]),
                ("set", [OBJECT_REMOVE_ADAPTER_NAME], [1.3]),
            ],
        )
        infers = [call for call in self.pipe.calls if call[0] == "infer"]
        self.assertEqual(
            infers,
            [
                ("infer", (CONSISTENCY_ADAPTER_NAME,), [1.0]),
                ("infer", (OBJECT_REMOVE_ADAPTER_NAME,), [1.3]),
            ],
        )

    def test_zero_strength_does_not_load_or_activate(self):
        generate_klein_edit(
            self.pipe, self.image, "add", self.config, lora_strength=0.0
        )
        self.assertEqual(self.pipe.loaded, {})
        infers = [call for call in self.pipe.calls if call[0] == "infer"]
        self.assertEqual(infers, [("infer", (), None)])

    def test_second_load_of_same_adapter_is_a_noop(self):
        generate_klein(
            self.pipe,
            self.image,
            "a",
            self.config,
            adapter_name="x",
            adapter_weight=1.0,
            pretrained="repo",
            weight_name="w.safetensors",
        )
        generate_klein(
            self.pipe,
            self.image,
            "b",
            self.config,
            adapter_name="x",
            adapter_weight=0.5,
            pretrained="repo",
            weight_name="w.safetensors",
        )
        loads = [call for call in self.pipe.calls if call[0] == "load"]
        self.assertEqual(len(loads), 1)

    def test_remove_lora_specs_use_distinct_adapters(self):
        fal = remove_lora_spec("fal")
        slopshop = remove_lora_spec("slopshop")
        self.assertEqual(fal.adapter_name, OBJECT_REMOVE_ADAPTER_NAME)
        self.assertEqual(slopshop.adapter_name, SLOPSHOP_ERASE_ADAPTER_NAME)
        self.assertNotEqual(fal.adapter_name, slopshop.adapter_name)
        self.assertTrue(fal.highlight)
        self.assertFalse(slopshop.highlight)

    def test_prompt_embeds_are_moved_to_the_pipeline_device(self):
        self.pipe.device = torch.device("cpu")
        self.pipe.transformer = SimpleNamespace(
            device=torch.device("cpu"), dtype=torch.float32
        )
        original = torch.zeros(1, 4, 8, dtype=torch.float16)
        generate_klein(
            self.pipe,
            self.image,
            None,
            self.config,
            prompt_embeds=original,
        )
        passed = next(
            call[1]["prompt_embeds"]
            for call in self.pipe.calls
            if call[0] == "infer_kwargs"
        )
        self.assertEqual(passed.dtype, torch.float32)
        self.assertEqual(passed.device.type, "cpu")
        self.assertEqual(original.dtype, torch.float16)

    def test_unknown_remove_lora_is_rejected(self):
        with self.assertRaises(ValueError):
            KleinInferenceConfig(remove_lora="unknown")
        with self.assertRaises(ValueError):
            remove_lora_spec("unknown")


if __name__ == "__main__":
    unittest.main()
