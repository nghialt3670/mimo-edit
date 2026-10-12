from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch

from evaluation.compbench_metrics import (
    _captions,
    _resize_mask,
    _score_pair,
    _summary,
    collect_saved_outputs,
)


class ScalarMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.arguments: tuple[torch.Tensor, ...] = ()

    def __call__(self, *arguments: torch.Tensor) -> torch.Tensor:
        self.arguments = arguments
        return torch.tensor(self.value)


class CompBenchMetricsTests(unittest.TestCase):
    def test_score_pair_uses_foreground_crop_and_masked_background(self) -> None:
        class Inputs(dict):
            def to(self, _device: torch.device):
                return self

        class Processor:
            def __init__(self) -> None:
                self.image_sizes: list[list[tuple[int, int]]] = []
                self.captions: list[str] = []

            def __call__(self, *, images, return_tensors, padding, text=None):
                self.image_sizes.append([image.size for image in images])
                input_ids = torch.empty((0, 1), dtype=torch.long)
                if text:
                    self.captions.extend(text)
                    input_ids = torch.tensor(
                        [[1 if caption == "red bird" else 2] for caption in text]
                    )
                return Inputs(
                    pixel_values=torch.empty((len(images), 3, 224, 224)),
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                )

        class ImageModel:
            @staticmethod
            def get_image_features(**inputs: torch.Tensor):
                batch_size = len(inputs["pixel_values"])
                return SimpleNamespace(
                    pooler_output=torch.tensor([[1.0, 0.0]] * batch_size)
                )

            @staticmethod
            def get_text_features(
                *, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
            ):
                del attention_mask
                features = [
                    [1.0, 0.0] if token == 1 else [0.0, 1.0]
                    for token in input_ids[:, 0].tolist()
                ]
                return SimpleNamespace(pooler_output=torch.tensor(features))

        psnr = ScalarMetric(25.0)
        ssim = ScalarMetric(0.8)
        lpips = ScalarMetric(0.1)
        processor = Processor()
        models = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "clip_processor": processor,
            "clip_model": ImageModel(),
        }
        mask = Image.new("L", (2, 2), 0)
        mask.putpixel((0, 0), 255)

        scores, bbox = _score_pair(
            Image.new("RGB", (2, 2), (64, 64, 64)),
            Image.new("RGB", (2, 2), (128, 128, 128)),
            Image.new("RGB", (2, 2), (90, 90, 90)),
            mask,
            ["red bird", "blue bird"],
            models=models,
            device=torch.device("cpu"),
        )

        self.assertEqual(bbox, (0, 0, 256, 256))
        self.assertEqual(processor.captions, ["red bird", "blue bird"])
        self.assertEqual(
            processor.image_sizes,
            [[(256, 256)], [(256, 256)], [(256, 256), (256, 256)]],
        )
        self.assertAlmostEqual(scores["LC-T"], 50.0)
        self.assertAlmostEqual(scores["LC-I"], 1.0)
        self.assertAlmostEqual(scores["PSNR"], 25.0)
        self.assertAlmostEqual(scores["SSIM"], 0.8)
        self.assertAlmostEqual(scores["LPIPS"], 0.1)
        background = psnr.arguments[0]
        self.assertEqual(background.shape, (1, 3, 512, 512))
        self.assertEqual(background[0, 0, 100, 100].item(), 0.0)
        self.assertAlmostEqual(background[0, 0, 400, 400].item(), 128 / 255)

    def test_pipe_separated_multi_object_captions_are_normalized(self) -> None:
        self.assertEqual(
            _captions(" a red bird |  a blue bird "),
            ["a red bird", "a blue bird"],
        )
        self.assertEqual(_captions([" one ", " ", "two"]), ["one", "two"])
        self.assertEqual(_captions(None), [])

    def test_mask_resize_is_binary_and_uses_nearest_neighbor(self) -> None:
        mask = Image.new("L", (2, 2), 0)
        mask.putpixel((0, 0), 255)
        resized = _resize_mask(mask)
        self.assertEqual(resized.size, (512, 512))
        self.assertEqual(set(resized.tobytes()), {0, 255})
        self.assertEqual(resized.getbbox(), (0, 0, 256, 256))

    def test_summary_includes_task_and_sample_weighted_overall_means(self) -> None:
        rows = [
            {
                "task": "add",
                "LC-T": 0.2,
                "LC-I": 0.4,
                "PSNR": 10.0,
                "SSIM": 0.5,
                "LPIPS": 0.3,
            },
            {
                "task": "remove",
                "LC-T": 0.6,
                "LC-I": 0.8,
                "PSNR": 20.0,
                "SSIM": 0.7,
                "LPIPS": 0.1,
            },
        ]
        summary = _summary(rows)
        self.assertEqual(summary["add"]["count"], 1)
        self.assertEqual(summary["overall"]["count"], 2)
        self.assertEqual(summary["overall"]["LC-T"], 0.4)
        self.assertEqual(summary["overall"]["PSNR"], 15.0)

    def test_collect_outputs_discovers_images_not_in_results_manifest(self) -> None:
        records = [{"sample_id": "add:sample.png"}]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            images_dir = run_dir / "results" / "images"
            images_dir.mkdir(parents=True)
            (run_dir / "results" / "results.jsonl").write_text("", encoding="utf-8")
            image_path = images_dir / "000000_add_sample.png_output.png"
            Image.new("RGB", (3, 3)).save(image_path)

            outputs = collect_saved_outputs(
                records, run_dir, prompt_mode="mixed"
            )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["record_position"], 0)
        self.assertEqual(outputs[0]["sample_id"], "add:sample.png")
        self.assertEqual(outputs[0]["output_path"], image_path.resolve())

    def test_collect_outputs_joins_sparse_dataset_index_by_sample_id(self) -> None:
        records = [
            {"sample_id": "add:first.png", "metadata": {"row_index": 0}},
            {
                "sample_id": "multi_object_add:second.png",
                "metadata": {"row_index": 2761},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            results_dir = run_dir / "results"
            images_dir = results_dir / "images"
            images_dir.mkdir(parents=True)
            image_path = images_dir / "002533_multi_object_add_second.png_output.png"
            Image.new("RGB", (3, 3)).save(image_path)
            (results_dir / "results.jsonl").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "annotation_index": 2761,
                        "sample_id": "multi_object_add:second.png",
                        "prompt_mode": "mixed",
                        "output_path": str(run_dir / "missing.png"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            outputs = collect_saved_outputs(records, run_dir, prompt_mode="mixed")

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["record_position"], 1)
        self.assertEqual(outputs[0]["sample_id"], "multi_object_add:second.png")
        self.assertEqual(outputs[0]["output_path"], image_path.resolve())

    def test_collect_outputs_rejects_result_for_another_prompt_mode(self) -> None:
        records = [{"sample_id": "add:sample.png"}]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            results_dir = run_dir / "results"
            results_dir.mkdir()
            result_path = run_dir / "generated.png"
            Image.new("RGB", (3, 3)).save(result_path)
            (results_dir / "results.jsonl").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "annotation_index": 0,
                        "sample_id": "add:sample.png",
                        "prompt_mode": "text",
                        "output_path": str(result_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Expected 'mixed'"):
                collect_saved_outputs(records, run_dir, prompt_mode="mixed")


if __name__ == "__main__":
    unittest.main()
