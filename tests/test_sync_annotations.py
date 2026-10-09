from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.annotation_targets import annotation_target
from scripts.sync_annotations import sync_annotations


class SyncAnnotationsTests(unittest.TestCase):
    def test_removes_resolved_audit_rows_and_sorts_successful_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.jsonl"
            audit = root / "audit.jsonl"
            records = [
                {"sample_id": "later", "metadata": {"row_index": 8}},
                {"sample_id": "earlier", "metadata": {"row_index": 2}},
            ]
            annotations.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            audit.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in [
                        {"sample_id": "later", "status": "failed"},
                        {"sample_id": "retry", "status": "failed"},
                        {"sample_id": None, "status": "failed"},
                    ]
                ),
                encoding="utf-8",
            )

            result = sync_annotations(annotations, audit, apply=True)

            sorted_records = [
                json.loads(line)
                for line in annotations.read_text(encoding="utf-8").splitlines()
            ]
            retained_audit = [
                json.loads(line)
                for line in audit.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result, {
            "annotations": 2,
            "sorted": True,
            "audit_removed": 1,
            "audit_retained": 2,
        })
        self.assertEqual([record["sample_id"] for record in sorted_records], ["earlier", "later"])
        self.assertEqual(
            [record["sample_id"] for record in retained_audit],
            ["retry", None],
        )

    def test_dry_run_reports_both_changes_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.jsonl"
            audit = root / "audit.jsonl"
            original_annotations = (
                json.dumps({"sample_id": "later", "metadata": {"row_index": 2}})
                + "\n"
                + json.dumps({"sample_id": "earlier", "metadata": {"row_index": 1}})
                + "\n"
            )
            original_audit = json.dumps({"sample_id": "later"}) + "\n"
            annotations.write_text(original_annotations, encoding="utf-8")
            audit.write_text(original_audit, encoding="utf-8")

            result = sync_annotations(annotations, audit)

            self.assertTrue(result["sorted"])
            self.assertEqual(result["audit_removed"], 1)
            self.assertEqual(annotations.read_text(encoding="utf-8"), original_annotations)
            self.assertEqual(audit.read_text(encoding="utf-8"), original_audit)

    def test_invalid_row_index_and_invalid_audit_json_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.jsonl"
            audit = root / "audit.jsonl"
            annotations.write_text(json.dumps({"metadata": {}}) + "\n", encoding="utf-8")
            audit.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata.row_index"):
                sync_annotations(annotations, audit, apply=True)

            annotations.write_text(
                json.dumps({"sample_id": "ok", "metadata": {"row_index": 0}}) + "\n",
                encoding="utf-8",
            )
            audit.write_text("{invalid json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                sync_annotations(annotations, audit, apply=True)

    def test_target_defaults_cover_each_supported_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.annotation_targets.load_project_environment"):
                compbench = annotation_target("compbench", root)
                magicbrush = annotation_target("magicbrush", root)
                refedit_bench = annotation_target("refedit_bench", root)

        self.assertEqual(
            compbench.annotations,
            root / ".out/annotation/compbench/compbench_local_multiobject_mixed.jsonl",
        )
        self.assertEqual(
            magicbrush.annotations,
            root / ".out/annotation/magicbrush/magicbrush_dev_mixed.jsonl",
        )
        self.assertEqual(
            refedit_bench.annotations,
            root / ".out/annotation/refedit_bench/refedit_bench_mixed.jsonl",
        )

    def test_target_respects_magicbrush_split_and_output_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("scripts.annotation_targets.load_project_environment"),
                patch.dict(
                    os.environ,
                    {
                        "MAGICBRUSH_SPLIT": "train",
                        "ANNOTATE_OUT": str(root / "custom"),
                    },
                    clear=True,
                ),
            ):
                target = annotation_target("magicbrush", root)

        self.assertEqual(target.annotations, root / "custom/magicbrush_train_mixed.jsonl")
        self.assertEqual(target.audit, root / "custom/magicbrush_train_audit.jsonl")

    def test_unknown_benchmark_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported benchmark"):
            annotation_target("unknown", Path("/tmp"))

    def test_invalid_magicbrush_split_is_rejected(self) -> None:
        with (
            patch("scripts.annotation_targets.load_project_environment"),
            patch.dict(os.environ, {"MAGICBRUSH_SPLIT": "../outside"}, clear=True),
        ):
            with self.assertRaisesRegex(ValueError, "MAGICBRUSH_SPLIT"):
                annotation_target("magicbrush", Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
