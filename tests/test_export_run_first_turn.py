import json
import tempfile
import unittest
from pathlib import Path

from evaluation.first_turn import (
    FIRST_TURN_FILE_NAME,
    FirstTurnStore,
    export_run_to_first_turn,
    task_fingerprint,
)


def _write_sample(run_dir: Path, stem: str, *, steps, raw_steps=None, model_id="m") -> None:
    log_dir = run_dir / "logs" / stem
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "trajectory.json").write_text(json.dumps(steps), encoding="utf-8")
    if raw_steps is not None:
        (log_dir / "raw.json").write_text(
            json.dumps({"model_id": model_id, "agent_raw_steps": raw_steps}),
            encoding="utf-8",
        )


class ExportRunFirstTurnTests(unittest.TestCase):
    def test_exports_first_act_and_round_trips_through_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_1"
            (run_dir / "results").mkdir(parents=True)
            results = [
                {
                    "sample_id": "s1",
                    "annotation_index": 0,
                    "prompt_mode": "mixed",
                    "instruction": "Make <box_0> red",
                    "reference_prompts": [{"type": "box", "variable": "box_0"}],
                    "log_dir": "/elsewhere/logs/002365_s1",
                    "status": "completed",
                },
                {
                    "sample_id": "s2",
                    "annotation_index": 1,
                    "prompt_mode": "text",
                    "instruction": "Remove the dog",
                    "reference_prompts": [],
                    "log_dir": str(run_dir / "logs" / "000001_s2"),
                    "status": "completed",
                },
            ]
            (run_dir / "results" / "results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8"
            )
            _write_sample(
                run_dir,
                "000000_s1",
                steps=[
                    {"observation": "task"},
                    {"thought": "Segment first.", "act": "x = segment_object(box_0)"},
                    {"act": "send_message(x)"},
                ],
                raw_steps=[
                    {"task": "task"},
                    {"model_output": "RAW-1"},
                ],
            )
            _write_sample(
                run_dir,
                "000001_s2",
                steps=[{"observation": "task"}],
            )

            path, exported, skipped = export_run_to_first_turn(run_dir)

            self.assertEqual(path, (run_dir / FIRST_TURN_FILE_NAME).resolve())
            self.assertEqual(exported, 1)
            self.assertEqual(skipped, ["s2"])

            store = FirstTurnStore(path)
            entry = store.get("s1", "mixed", "box")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.task, "Make <box_0> red")
            self.assertEqual(entry.objects, ["box_0", "image_0"])
            self.assertEqual(
                entry.fingerprint,
                task_fingerprint("Make <box_0> red", {"image_0": None, "box_0": None}),
            )
            self.assertEqual(entry.thought, "Segment first.")
            self.assertEqual(entry.code, "x = segment_object(box_0)")
            self.assertEqual(entry.raw_output, "RAW-1")
            self.assertEqual(entry.model_id, "m")

    def test_missing_raw_json_falls_back_to_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2"
            (run_dir / "results").mkdir(parents=True)
            record = {
                "sample_id": "s3",
                "annotation_index": 0,
                "prompt_mode": "text",
                "instruction": "Add a hat",
                "reference_prompts": [],
                "log_dir": "/elsewhere/logs/000000_s3",
            }
            (run_dir / "results" / "results.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            _write_sample(run_dir, "000000_s3", steps=[{"act": "send_message(image_0)"}])

            _, exported, _ = export_run_to_first_turn(run_dir, Path(tmp) / "out.jsonl")

            self.assertEqual(exported, 1)
            entry = FirstTurnStore(Path(tmp) / "out.jsonl").get("s3", "text", "")
            self.assertEqual(entry.raw_output, "send_message(image_0)")
            self.assertIsNone(entry.model_id)

    def test_missing_results_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                export_run_to_first_turn(Path(tmp))


if __name__ == "__main__":
    unittest.main()
