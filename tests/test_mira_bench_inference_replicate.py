from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.mira_bench_inference_replicate import (
    _read_rows,
    make_parser,
    prepare_samples,
    run_benchmark,
)


class MiraBenchReplicateInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "dataset" / "benchmark"
        self.dataset_root.mkdir(parents=True)
        Image.new("RGB", (8, 6), "red").save(self.dataset_root / "one.png")
        Image.new("RGB", (8, 6), "blue").save(self.dataset_root / "two.png")
        self.rows = [
            {"image": "one.png", "editing_instruction": "Edit the first."},
            {"image": "two.png", "editing_instruction": "Edit the second."},
        ]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_selects_requested_sample_ids_and_caches_sources(self) -> None:
        samples = prepare_samples(
            self.root / "run",
            self.rows,
            dataset_root=self.dataset_root,
            sample_ids=["mirage:two.png"],
        )
        self.assertEqual([sample["id"] for sample in samples], ["mirage:two.png"])
        self.assertEqual(samples[0]["instruction"], "Edit the second.")
        self.assertTrue((self.root / "run" / samples[0]["input_image"]).is_file())
        self.assertTrue((self.root / "run" / "selection.jsonl").is_file())

    def test_unknown_sample_id_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown MIRA-Bench sample ID"):
            prepare_samples(
                self.root / "run",
                self.rows,
                dataset_root=self.dataset_root,
                sample_ids=["mirage:missing.png"],
            )

    def test_selects_zero_based_annotation_indices_in_requested_order(self) -> None:
        samples = prepare_samples(
            self.root / "run",
            self.rows,
            dataset_root=self.dataset_root,
            sample_indices=[1, 0],
        )
        self.assertEqual(
            [sample["id"] for sample in samples],
            ["mirage:two.png", "mirage:one.png"],
        )

    def test_out_of_range_annotation_index_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            prepare_samples(
                self.root / "run",
                self.rows,
                dataset_root=self.dataset_root,
                sample_indices=[2],
            )

    def test_dry_run_writes_timestamped_run_under_output_root(self) -> None:
        dataset_root = self.root / "local_dataset"
        benchmark_root = dataset_root / "benchmark"
        benchmark_root.mkdir(parents=True)
        image_path = benchmark_root / "one.png"
        Image.new("RGB", (8, 6), "red").save(image_path)
        (benchmark_root / "annotations.jsonl").write_text(
            json.dumps(self.rows[0]) + "\n", encoding="utf-8"
        )
        output_root = self.root / ".out" / "evaluation" / "mira_bench"
        args = make_parser().parse_args(
            [
                "--output-root",
                str(output_root),
                "--dataset-dir",
                str(dataset_root),
                "--sample-ids",
                "mirage:one.png",
                "--models",
                "bagel",
                "--dry-run",
            ]
        )

        summary = run_benchmark(args)

        run_dirs = list(output_root.glob("run_*"))
        self.assertEqual(len(run_dirs), 1)
        self.assertEqual(summary["bagel"]["not_started"], 1)
        selection = _read_rows(run_dirs[0] / "selection.jsonl")
        self.assertEqual(selection[0]["id"], "mirage:one.png")

    def test_parser_accepts_multiple_sample_ids(self) -> None:
        args = make_parser().parse_args(
            [
                "--sample-ids",
                "mirage:one.png",
                "mirage:two.png",
                "--dry-run",
            ]
        )
        self.assertEqual(
            args.sample_ids,
            ["mirage:one.png", "mirage:two.png"],
        )

    def test_parser_accepts_multiple_sample_indices(self) -> None:
        args = make_parser().parse_args(
            ["--sample-indices", "10", "30", "4", "57", "--dry-run"]
        )
        self.assertEqual(args.sample_indices, [10, 30, 4, 57])


if __name__ == "__main__":
    unittest.main()
