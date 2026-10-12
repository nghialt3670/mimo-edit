from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from annotation.runner import convert_rows
from annotation.schema import AnnotationRecord


class _Annotator:
    def annotate(self, sample):
        return AnnotationRecord(
            sample_id=sample.sample_id,
            benchmark="CompBench",
            split="train",
            original_instruction="updated instruction",
            mixed_instruction="updated instruction",
            references=[],
        )


class _FailingAnnotator:
    def annotate(self, sample):
        raise RuntimeError("parser unavailable")


class AnnotationRunnerOverwriteTests(unittest.TestCase):
    def test_overwrite_replaces_only_selected_existing_sample(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "annotations.jsonl"
            failures = Path(directory) / "audit.jsonl"
            output.write_text(
                "\n".join(
                    (
                        json.dumps({"sample_id": "sample-a", "value": "old"}),
                        json.dumps({"sample_id": "sample-b", "value": "keep"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            counts = convert_rows(
                rows=[{"sample_id": "sample-a"}],
                adapter=lambda row, split, index: SimpleNamespace(
                    sample_id=row["sample_id"],
                    instruction="instruction",
                ),
                split="train",
                annotator=_Annotator(),
                output=output,
                failures=failures,
                save_assets=False,
                overwrite_sample_ids={"sample-a"},
            )

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(counts["accepted"], 1)
            self.assertEqual(
                [record["sample_id"] for record in records],
                ["sample-a", "sample-b"],
            )
            self.assertEqual(
                records[0]["original_instruction"], "updated instruction"
            )
            self.assertEqual(records[1]["value"], "keep")

    def test_failed_overwrite_keeps_the_previous_annotation(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "annotations.jsonl"
            failures = Path(directory) / "audit.jsonl"
            original = {
                "sample_id": "sample-a",
                "original_instruction": "old accepted result",
            }
            output.write_text(json.dumps(original) + "\n", encoding="utf-8")

            counts = convert_rows(
                rows=[{"sample_id": "sample-a"}],
                adapter=lambda row, split, index: SimpleNamespace(
                    sample_id=row["sample_id"],
                    instruction="instruction",
                ),
                split="train",
                annotator=_FailingAnnotator(),
                output=output,
                failures=failures,
                save_assets=False,
                overwrite_sample_ids={"sample-a"},
            )

            self.assertEqual(counts["failed"], 1)
            self.assertEqual(json.loads(output.read_text()), original)
            audit_record = json.loads(failures.read_text())
            self.assertEqual(audit_record["sample_id"], "sample-a")


if __name__ == "__main__":
    unittest.main()
