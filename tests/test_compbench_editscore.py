from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from evaluation.compbench_editscore import (
    _enable_editscore_mps,
    evaluate_replicate_compbench_outputs,
    evaluate_saved_compbench_outputs,
)


class FakeEditScore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Image.Image], str]] = []

    def evaluate(self, images: list[Image.Image], instruction: str) -> dict[str, float]:
        self.calls.append((images, instruction))
        return {
            "prompt_following": 8.1,
            "consistency": 7.2,
            "perceptual_quality": 6.3,
            "overall": 7.1,
        }


class FakeImageLoader:
    def __init__(self) -> None:
        self.row = {
            "task": "multi_object_add",
            "image_path": "multi.png",
            "instruction": "add a red bird",
            "input_image": Image.new("RGB", (10, 8), "white"),
        }

    @staticmethod
    def parse_reference(_reference: str) -> tuple[str, str, int, str]:
        return "dataset", "train", 0, "input_image"

    def get_row(self, *_args: Any) -> dict[str, Any]:
        return self.row

    @staticmethod
    def as_image(image: Image.Image) -> Image.Image:
        return image


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _write_standard_outputs(output_dir: Path) -> list[dict[str, Any]]:
    results_dir = output_dir / "results"
    images_dir = results_dir / "images"
    images_dir.mkdir(parents=True)
    output_path = images_dir / "multi_object_add_multi.png_output.png"
    Image.new("RGB", (12, 9), "red").save(output_path)
    ignored_path = images_dir / "add_single.png_output.png"
    Image.new("RGB", (10, 8), "blue").save(ignored_path)
    records = [
        {
            "sample_id": "multi_object_add:multi.png",
            "original_instruction": "add a red bird",
            "source_ref": "hf://dataset/train/0/input_image",
            "metadata": {
                "task": "multi_object_add",
                "row_index": 0,
                "image_path": "multi.png",
            },
        },
        {
            "sample_id": "add:single.png",
            "original_instruction": "add a red bird",
            "source_ref": "hf://dataset/train/1/input_image",
            "metadata": {
                "task": "add",
                "row_index": 1,
                "image_path": "single.png",
            },
        },
    ]
    results = [
        {
            "status": "completed",
            "sample_id": record["sample_id"],
            "prompt_mode": "mixed",
            "output_path": str(path),
        }
        for record, path in zip(records, (output_path, ignored_path), strict=True)
    ]
    (results_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
    )
    return records


class CompBenchEditScoreTests(unittest.TestCase):
    def _fake_editscore_adapter(self, module_name: str, class_name: str):
        calls: dict[str, Any] = {}

        class FakeBatch:
            def to(self, device: str) -> FakeBatch:
                calls["input_device"] = device
                return self

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeModel:
                calls["model_args"] = args
                calls["model_kwargs"] = kwargs
                return cls()

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, model_name: str) -> FakeProcessor:
                calls["processor_model"] = model_name
                return cls()

            def apply_chat_template(self, *args: Any, **kwargs: Any) -> FakeBatch:
                calls["template_args"] = args
                calls["template_kwargs"] = kwargs
                return FakeBatch()

            def __call__(self, **kwargs: Any) -> FakeBatch:
                calls["processor_args"] = kwargs
                return FakeBatch()

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model: FakeModel, lora_path: str) -> FakePeftModel:
                calls["lora_path"] = lora_path
                return cls()

            def merge_and_unload(self) -> FakeModel:
                return FakeModel()

        module = types.ModuleType(module_name)
        setattr(module, class_name, type(class_name, (), {}))
        setattr(
            module,
            (
                "Qwen3VLForConditionalGeneration"
                if class_name == "Qwen3VL"
                else "Qwen2_5_VLForConditionalGeneration"
            ),
            FakeModel,
        )
        module.AutoProcessor = FakeProcessor
        module.PeftModel = FakePeftModel
        module.apply_chat_template = lambda prompt, num_images: f"{prompt}:{num_images}"
        module.process_vision_info = lambda _messages: (["image"], [])
        return module, calls

    def test_mps_adapter_places_qwen3_weights_and_inputs_on_mps(self) -> None:
        module_name = "editscore.mllm_tools.qwen3vl"
        module, calls = self._fake_editscore_adapter(module_name, "Qwen3VL")
        package = types.ModuleType("editscore")
        package.__path__ = []
        adapters = types.ModuleType("editscore.mllm_tools")
        adapters.__path__ = []
        with (
            unittest.mock.patch.dict(
                sys.modules,
                {
                    "editscore": package,
                    "editscore.mllm_tools": adapters,
                    module_name: module,
                },
            ),
            unittest.mock.patch("torch.backends.mps.is_available", return_value=True),
            unittest.mock.patch("torch.cuda.is_available", return_value=False),
        ):
            self.assertTrue(_enable_editscore_mps("qwen3vl"))
            scorer = module.Qwen3VL(
                "base-model", temperature=0.2, seed=4, lora_path="adapter"
            )
            result = scorer.prepare_input([Image.new("RGB", (2, 2))], "prompt")

        self.assertEqual(
            calls["model_kwargs"]["torch_dtype"], __import__("torch").float16
        )
        self.assertEqual(calls["model_kwargs"]["device_map"], "mps")
        self.assertEqual(calls["input_device"], "mps")
        self.assertEqual(calls["lora_path"], "adapter")
        self.assertIsNotNone(result)

    def test_mps_adapter_supports_qwen25_vision_preprocessing(self) -> None:
        module_name = "editscore.mllm_tools.qwen25vl"
        module, calls = self._fake_editscore_adapter(module_name, "Qwen25VL")
        package = types.ModuleType("editscore")
        package.__path__ = []
        adapters = types.ModuleType("editscore.mllm_tools")
        adapters.__path__ = []
        with (
            unittest.mock.patch.dict(
                sys.modules,
                {
                    "editscore": package,
                    "editscore.mllm_tools": adapters,
                    module_name: module,
                },
            ),
            unittest.mock.patch("torch.backends.mps.is_available", return_value=True),
            unittest.mock.patch("torch.cuda.is_available", return_value=False),
        ):
            self.assertTrue(_enable_editscore_mps("qwen25vl"))
            scorer = module.Qwen25VL("base-model")
            scorer.prepare_input([Image.new("RGB", (2, 2))], "prompt")

        self.assertEqual(calls["input_device"], "mps")
        self.assertEqual(calls["processor_args"]["videos"], [])
        self.assertEqual(calls["processor_args"]["images"], ["image"])

    def test_mps_device_request_fails_when_unavailable(self) -> None:
        with (
            unittest.mock.patch.dict("os.environ", {"EDITSCORE_DEVICE": "mps"}),
            unittest.mock.patch("torch.backends.mps.is_available", return_value=False),
            self.assertRaisesRegex(EnvironmentError, "no available MPS device"),
        ):
            _enable_editscore_mps("qwen3vl")

    def test_standard_scores_only_multi_object_outputs_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "run"
            records = _write_standard_outputs(output_dir)
            scorer = FakeEditScore()

            rows, summary = evaluate_saved_compbench_outputs(
                records,
                output_dir=output_dir,
                image_loader=FakeImageLoader(),
                prompt_mode="mixed",
                scorer=scorer,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task"], "multi_object_add")
            self.assertEqual(rows[0]["overall"], 7.1)
            self.assertEqual(scorer.calls[0][1], "add a red bird")
            self.assertEqual(scorer.calls[0][0][0].size, (10, 8))
            self.assertEqual(scorer.calls[0][0][1].size, (10, 8))
            self.assertEqual(summary["overall"]["count"], 1)
            self.assertEqual(summary["by_task"]["multi_object_add"]["overall"], 7.1)

            evaluate_saved_compbench_outputs(
                records,
                output_dir=output_dir,
                image_loader=FakeImageLoader(),
                prompt_mode="mixed",
                scorer=scorer,
            )
            self.assertEqual(len(scorer.calls), 1)
            metric_path = output_dir / "metrics" / "editscore.jsonl"
            self.assertTrue(metric_path.is_file())
            self.assertTrue(
                (output_dir / "metrics" / "editscore_summary.json").is_file()
            )

    def test_replicate_scores_and_summarizes_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.png"
            mask_path = root / "mask.png"
            output_path = root / "generated.png"
            Image.new("RGB", (10, 8), "white").save(source_path)
            Image.new("L", (10, 8), 255).save(mask_path)
            Image.new("RGB", (12, 9), "green").save(output_path)
            sample = {
                "id": "sample-id",
                "task": "multi_object_add",
                "image_path": "sample.png",
                "instruction": "add a red bird",
                "input_image": "source.png",
                "mask": mask_path.name,
            }
            (root / "selection.jsonl").write_text(
                json.dumps(sample) + "\n", encoding="utf-8"
            )
            _write_json(
                root / "records" / "bagel" / "multi_object_add" / "sample.json",
                {
                    "sample_id": "sample-id",
                    "task": "multi_object_add",
                    "image_path": "sample.png",
                    "model": "bagel",
                    "prompt": "add a red bird",
                    "status": "succeeded",
                    "output_path": "generated.png",
                },
            )
            scorer = FakeEditScore()

            rows, summary = evaluate_replicate_compbench_outputs(root, scorer=scorer)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "bagel")
            self.assertEqual(rows[0]["task"], "multi_object_add")
            self.assertEqual(summary["by_model"]["bagel"]["overall"]["count"], 1)
            self.assertEqual(
                summary["by_model"]["bagel"]["by_task"]["multi_object_add"]["overall"],
                7.1,
            )
            self.assertEqual(scorer.calls[0][0][0].size, (10, 8))
            self.assertEqual(scorer.calls[0][0][1].size, (10, 8))


if __name__ == "__main__":
    unittest.main()
