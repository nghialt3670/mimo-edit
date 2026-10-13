from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.compbench_incomplete import (
    find_incomplete_compbench_samples,
    read_sample_indices,
    write_sample_indices,
)


class CompBenchIncompleteTests(unittest.TestCase):
    def test_finds_missing_failed_and_missing_output_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            valid_output = run_dir / "valid.png"
            valid_output.write_bytes(b"image")
            renamed_output = run_dir / "results" / "images" / (
                "002800_add_renamed.png_output.png"
            )
            renamed_output.parent.mkdir(parents=True)
            renamed_output.write_bytes(b"image")
            records = [
                {"sample_id": "add:missing.png", "metadata": {"row_index": 10}},
                {"sample_id": "remove:failed.png", "metadata": {"row_index": 42}},
                {
                    "sample_id": "replace:missing-output.png",
                    "metadata": {"row_index": 2761},
                },
                {"sample_id": "add:valid.png", "metadata": {"row_index": 2904}},
                {"sample_id": "add:renamed.png", "metadata": {"row_index": 2800}},
            ]
            results = [
                {
                    "annotation_index": 42,
                    "sample_id": "remove:failed.png",
                    "status": "failed",
                },
                {
                    "annotation_index": 2761,
                    "sample_id": "replace:missing-output.png",
                    "status": "completed",
                    "output_path": str(run_dir / "missing.png"),
                },
                {
                    "annotation_index": 2904,
                    "sample_id": "add:valid.png",
                    "status": "completed",
                    "output_path": str(valid_output),
                },
                {
                    "annotation_index": 2800,
                    "sample_id": "add:renamed.png",
                    "status": "completed",
                    "output_path": str(run_dir / "results" / "images" / "old_name.png"),
                },
            ]

            incomplete = find_incomplete_compbench_samples(
                records, results, run_dir
            )

        self.assertEqual(
            incomplete,
            [(0, "missing_result"), (1, "status_failed"), (2, "missing_output_file")],
        )

    def test_writes_annotated_indices_accepted_by_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retry.txt"
            records = [
                {"sample_id": "add:one.png", "metadata": {"row_index": 58}},
                {"sample_id": "remove:two.png", "metadata": {"row_index": 2904}},
            ]
            write_sample_indices(path, [(1, "status_failed")], records)

            indices = read_sample_indices(path, records)

            self.assertEqual(indices, [1])
            contents = path.read_text(encoding="utf-8")
            self.assertIn("2904 # status_failed", contents)
            self.assertIn("sample_id=remove:two.png", contents)

    def test_rejects_duplicate_and_unknown_row_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retry.txt"
            records = [
                {"metadata": {"row_index": 10}},
                {"metadata": {"row_index": 42}},
            ]
            path.write_text("42\n42\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate CompBench row index"):
                read_sample_indices(path, records)

            path.write_text("2761\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not present in the annotation file"):
                read_sample_indices(path, records)

    def test_rejects_empty_sample_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retry.txt"
            path.write_text("# no pending samples\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No CompBench row indices"):
                read_sample_indices(path, [])


if __name__ == "__main__":
    unittest.main()
