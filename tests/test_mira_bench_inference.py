from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from evaluation.common import (
    MiraBenchImageLoader,
    load_sample,
    select_reference_prompts,
)
from evaluation.notebook import load_benchmark


class MiraBenchImageLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_root = Path(self.temporary_directory.name) / "benchmark"
        self.image_root.mkdir()
        Image.new("RGB", (12, 8), color="red").save(self.image_root / "source.png")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_current_and_legacy_source_references(self):
        loader = MiraBenchImageLoader(self.image_root)
        for reference in (
            "mira_bench://benchmark/source.png",
            "hf://ziqiangoodgood/MIRAGE/benchmark/source.png",
        ):
            with self.subTest(reference=reference):
                image, target = loader.load_pair({"source_ref": reference})
                self.assertEqual(image.size, (12, 8))
                self.assertIsNone(target)

    def test_rejects_references_outside_dataset_root(self):
        loader = MiraBenchImageLoader(self.image_root)
        with self.assertRaisesRegex(ValueError, "escapes"):
            loader.load_pair({"source_ref": "mira_bench://benchmark/../outside.png"})

    def test_loads_symlinked_snapshot_files(self):
        blobs = Path(self.temporary_directory.name) / "blobs"
        blobs.mkdir()
        Image.new("RGB", (5, 5)).save(blobs / "blob.png", format="PNG")
        (self.image_root / "linked.png").symlink_to(blobs / "blob.png")
        image, _ = MiraBenchImageLoader(self.image_root).load_pair(
            {"source_ref": "mira_bench://benchmark/linked.png"}
        )
        self.assertEqual(image.size, (5, 5))

    def test_load_sample_uses_mode_specific_instructions(self):
        loader = MiraBenchImageLoader(self.image_root)
        record = {
            "sample_id": "mirage:source.png",
            "source_ref": "mira_bench://benchmark/source.png",
            "original_instruction": "Text instruction",
            "mixed_instruction": "Mixed <box_0>",
            "references": [
                {
                    "token": "<box_0>",
                    "span": "the object",
                    "role": "target",
                    "region": {"bbox_xyxy": [1, 1, 5, 5]},
                }
            ],
        }
        text_sample = load_sample(
            record, benchmark="mira_bench", image_loader=loader, prompt_mode="text"
        )
        mixed_sample = load_sample(
            record, benchmark="mira_bench", image_loader=loader, prompt_mode="mixed"
        )
        self.assertEqual(text_sample["instruction"], "Text instruction")
        self.assertEqual(mixed_sample["instruction"], "Mixed <box_0>")
        self.assertIsNone(text_sample["target_image"])
        with patch("evaluation.common.seed_random_generators", return_value=1):
            select_reference_prompts(text_sample, prompt_mode="text")
            select_reference_prompts(mixed_sample, prompt_mode="mixed")
        self.assertIsNone(text_sample["references"][0]["selected_prompt"])
        self.assertIsNotNone(mixed_sample["references"][0]["selected_prompt"])


class MiraBenchBenchmarkResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.image_root = self.project_root / "dataset" / "benchmark"
        self.image_root.mkdir(parents=True)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _load(self):
        with (
            patch(
                "evaluation.notebook.load_project_environment",
                return_value=self.project_root,
            ),
            patch("evaluation.notebook.seed_random_generators"),
        ):
            return load_benchmark("mira_bench", prompt_mode="mixed")

    def test_loads_latest_legacy_annotation_run(self):
        legacy_run = (
            self.project_root
            / ".out"
            / "annotation"
            / "mirage"
            / "run_20260101_000000_000000"
        )
        legacy_run.mkdir(parents=True)
        (legacy_run / "mirage_mixed.jsonl").write_text(
            json.dumps(
                {
                    "sample_id": "mirage:source.png",
                    "source_ref": "hf://ziqiangoodgood/MIRAGE/benchmark/source.png",
                    "original_instruction": "Text instruction",
                    "mixed_instruction": "Mixed instruction",
                    "references": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with patch.dict(
            os.environ,
            {"MIRA_BENCH_IMAGE_ROOT": str(self.image_root)},
            clear=False,
        ):
            for name in (
                "MMEDIT_MIRA_BENCH_ANNOTATIONS",
                "MMEDIT_MIRA_BENCH_ANNOTATION_RUN",
                "MMEDIT_MIRA_BENCH_OUTPUT_DIR",
                "ANNOTATE_OUT",
            ):
                os.environ.pop(name, None)
            context = self._load()
        self.assertEqual(context.annotation_file, legacy_run / "mirage_mixed.jsonl")
        self.assertEqual(
            context.output_dir,
            (
                self.project_root / ".out" / "evaluation" / "mira_bench" / "mixed"
            ).resolve(),
        )
        self.assertIsInstance(context.image_loader, MiraBenchImageLoader)
        self.assertEqual(len(context.records), 1)


if __name__ == "__main__":
    unittest.main()
