from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from evaluation.compbench_metrics import collect_saved_outputs
from evaluation.compbench_viescore import (
    collect_replicate_saved_outputs,
    evaluate_replicate_compbench_outputs,
    evaluate_saved_compbench_outputs,
)


class FakeScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Image.Image], str]] = []

    def evaluate(
        self,
        images: list[Image.Image],
        instruction: str,
        *,
        extract_all_score: bool,
    ) -> tuple[float, float, float]:
        assert extract_all_score
        self.calls.append((images, instruction))
        return 0.8, 0.7, 0.6


class FakeImageLoader:
    def __init__(self) -> None:
        self.row = {
            "task": "multi_object_add",
            "image_path": "multi.png",
            "instruction": "add a red bird",
            "input_image": Image.new("RGB", (10, 8), "white"),
            "mask": self._mask(),
        }

    @staticmethod
    def _mask() -> Image.Image:
        mask = Image.new("L", (10, 8), 0)
        ImageDraw.Draw(mask).rectangle((2, 1, 6, 5), fill=255)
        return mask

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


class CompBenchVIEScoreTests(unittest.TestCase):
    def test_standard_evaluation_scores_only_multi_object_task_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "run"
            results_dir = output_dir / "results"
            images_dir = results_dir / "images"
            images_dir.mkdir(parents=True)
            scored_output = images_dir / "add_sample_output.png"
            ignored_output = images_dir / "attribute_sample_output.png"
            Image.new("RGB", (10, 8), "red").save(scored_output)
            Image.new("RGB", (10, 8), "blue").save(ignored_output)
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
                    "sample_id": "add:single-object.png",
                    "original_instruction": "add one red bird",
                    "source_ref": "hf://dataset/train/1/input_image",
                    "metadata": {
                        "task": "add",
                        "row_index": 1,
                        "image_path": "single-object.png",
                    },
                },
                {
                    "sample_id": "attribute:sample.png",
                    "original_instruction": "make the bird blue",
                    "source_ref": "hf://dataset/train/0/input_image",
                    "metadata": {
                        "task": "attribute",
                        "row_index": 0,
                        "image_path": "sample.png",
                    },
                },
            ]
            results = [
                {
                    "status": "completed",
                    "sample_id": record["sample_id"],
                    "prompt_mode": "mixed",
                    "output_path": str(image_path),
                }
                for record, image_path in zip(
                    records,
                    (scored_output, ignored_output, ignored_output),
                    strict=True,
                )
            ]
            (results_dir / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in results),
                encoding="utf-8",
            )
            scorer = FakeScorer()

            filtered_outputs = collect_saved_outputs(
                records,
                output_dir,
                prompt_mode="mixed",
                task_filter={"multi_object_add", "multi_object_remove"},
            )
            self.assertEqual(len(filtered_outputs), 1)
            self.assertEqual(
                filtered_outputs[0]["sample_id"], "multi_object_add:multi.png"
            )

            rows, summary = evaluate_saved_compbench_outputs(
                records,
                output_dir=output_dir,
                image_loader=FakeImageLoader(),
                prompt_mode="mixed",
                scorer=scorer,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task"], "multi_object_add")
            self.assertEqual(rows[0]["roi_bbox_xyxy"], [2, 1, 7, 6])
            self.assertEqual(scorer.calls[0][1], "add a red bird")
            self.assertEqual(scorer.calls[0][0][0].size, (5, 5))
            self.assertEqual(summary["overall"]["count"], 1)
            self.assertEqual(summary["by_task"]["multi_object_add"]["sc"], 0.8)
            evaluate_saved_compbench_outputs(
                records,
                output_dir=output_dir,
                image_loader=FakeImageLoader(),
                prompt_mode="mixed",
                scorer=scorer,
            )
            self.assertEqual(len(scorer.calls), 1)

    def test_replicate_evaluation_uses_local_selection_and_model_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.png"
            mask_path = root / "mask.png"
            output_path = root / "generated.png"
            Image.new("RGB", (10, 8), "white").save(source_path)
            mask = Image.new("L", (10, 8), 0)
            ImageDraw.Draw(mask).rectangle((2, 1, 6, 5), fill=255)
            mask.save(mask_path)
            Image.new("RGB", (10, 8), "green").save(output_path)
            sample = {
                "id": "sample-id",
                "task": "multi_object_add",
                "image_path": "sample.png",
                "instruction": "add a red bird",
                "input_image": "source.png",
                "mask": "mask.png",
            }
            (root / "selection.jsonl").write_text(
                json.dumps(sample) + "\n", encoding="utf-8"
            )
            record_path = (
                root / "records" / "bagel" / "multi_object_add" / "sample.json"
            )
            _write_json(
                record_path,
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

            self.assertEqual(len(collect_replicate_saved_outputs(root)), 1)
            scorer = FakeScorer()
            rows, summary = evaluate_replicate_compbench_outputs(root, scorer=scorer)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "bagel")
            self.assertEqual(rows[0]["task"], "multi_object_add")
            self.assertEqual(rows[0]["roi_bbox_xyxy"], [2, 1, 7, 6])
            self.assertEqual(summary["by_model"]["bagel"]["overall"]["count"], 1)
            self.assertEqual(
                summary["by_model"]["bagel"]["by_task"]["multi_object_add"]["pq"],
                0.7,
            )

    def test_replicate_collection_rejects_paths_outside_result_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (4, 4), "white").save(root / "generated.png")
            sample = {
                "id": "sample-id",
                "task": "multi_object_add",
                "image_path": "sample.png",
                "instruction": "add a red bird",
                "input_image": "../source.png",
                "mask": "mask.png",
            }
            (root / "selection.jsonl").write_text(
                json.dumps(sample) + "\n", encoding="utf-8"
            )
            record_path = (
                root / "records" / "bagel" / "multi_object_add" / "sample.json"
            )
            _write_json(
                record_path,
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

            with self.assertRaisesRegex(ValueError, "escapes result root"):
                collect_replicate_saved_outputs(root)


if __name__ == "__main__":
    unittest.main()
