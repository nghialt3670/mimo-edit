from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.common import run_benchmark


class InferenceRerunCleanupTests(unittest.TestCase):
    def test_rerun_clears_only_selected_sample_logs_and_images(self) -> None:
        records = [{"sample_id": "sample:0"}, {"sample_id": "sample:1"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            logs_dir = run_dir / "logs"
            images_dir = run_dir / "results" / "images"
            selected_stem = "000001_sample_1"
            untouched_stem = "000000_sample_0"
            old_log = logs_dir / selected_stem
            old_debug_asset = old_log / "debug_assets" / "stale.png"
            old_debug_asset.parent.mkdir(parents=True)
            old_debug_asset.write_text("old", encoding="utf-8")
            untouched_log = logs_dir / untouched_stem
            untouched_log.mkdir(parents=True)
            (untouched_log / "raw.json").write_text("keep", encoding="utf-8")
            images_dir.mkdir(parents=True)
            for suffix in ("source", "annotated", "output", "target"):
                (images_dir / f"{selected_stem}_{suffix}.png").write_text(
                    "stale", encoding="utf-8"
                )
            untouched_image = images_dir / f"{untouched_stem}_output.png"
            untouched_image.write_text("keep", encoding="utf-8")

            results_path = run_dir / "results" / "results.jsonl"
            results_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "benchmark": "compbench",
                            "prompt_mode": "mixed",
                            "sample_id": record["sample_id"],
                            "annotation_index": index,
                            "status": "completed",
                            "log_dir": str(logs_dir / (
                                untouched_stem if index == 0 else selected_stem
                            )),
                        }
                    )
                    + "\n"
                    for index, record in enumerate(records)
                ),
                encoding="utf-8",
            )

            def fake_run_one(record, annotation_index, **kwargs):
                self.assertEqual(annotation_index, 1)
                in_progress_rows = [
                    json.loads(line)
                    for line in results_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(in_progress_rows[1]["status"], "running")
                self.assertNotIn("output_path", in_progress_rows[1])
                self.assertFalse(old_log.exists())
                self.assertFalse(old_debug_asset.exists())
                self.assertTrue(untouched_log.exists())
                self.assertTrue(untouched_image.exists())
                for suffix in ("source", "annotated", "output", "target"):
                    self.assertFalse(
                        (images_dir / f"{selected_stem}_{suffix}.png").exists()
                    )
                return {
                    "benchmark": "compbench",
                    "prompt_mode": "mixed",
                    "sample_id": record["sample_id"],
                    "annotation_index": annotation_index,
                    "status": "failed",
                    "elapsed_seconds": 1.0,
                    "log_dir": str(logs_dir / selected_stem),
                }

            with patch("evaluation.common.run_one", side_effect=fake_run_one):
                results = run_benchmark(
                    records,
                    [1],
                    benchmark="compbench",
                    prompt_mode="mixed",
                    output_dir=run_dir,
                    image_loader=object(),
                    model=object(),
                    tools=[],
                    resume=True,
                    rerun_completed=True,
                )

            self.assertEqual([result["status"] for result in results], ["failed"])
            saved_rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["status"] for row in saved_rows], ["completed", "failed"])


if __name__ == "__main__":
    unittest.main()
