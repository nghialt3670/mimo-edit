from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.first_turn import FirstTurnEntry, FirstTurnStore
from evaluation.first_turn_batch import (
    FirstTurnRequest,
    collect_results,
    load_manifest,
    parse_batch_output,
    write_batch_inputs,
)
from framework.core.agent import FirstTurnCommand


class _StubParser:
    """Parses the raw content as the code itself; thought is fixed."""

    def parse_first_turn_output(self, raw: str) -> FirstTurnCommand:
        return FirstTurnCommand(code=raw, thought="t", raw_output=raw)


GOOD_CODE = 'send_message(text="done", objects=[])'
BAD_CODE = "x = 1"


def _request(custom_id: str, sample_id: str) -> FirstTurnRequest:
    return FirstTurnRequest(
        custom_id=custom_id,
        sample_id=sample_id,
        prompt_mode="mixed",
        prompt_signature="sig",
        fingerprint="fp",
        task="task",
        objects=["image_0"],
        body={"model": "m", "messages": []},
    )


def _ok_line(custom_id: str, content: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": content}}]},
            },
            "error": None,
        }
    )


class FirstTurnBatchTests(unittest.TestCase):
    def test_write_batch_inputs_writes_jsonl_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "batch"
            write_batch_inputs(
                work_dir,
                [_request("000001", "s1")],
                benchmark="refedit_bench",
                model_id="gemini-3.1-flash-lite",
                store_path=Path(tmp) / "first_turn.jsonl",
            )
            line = json.loads((work_dir / "requests.jsonl").read_text().splitlines()[0])
            self.assertEqual(line["custom_id"], "000001")
            self.assertEqual(line["method"], "POST")
            self.assertEqual(line["url"], "/v1/chat/completions")
            manifest = load_manifest(work_dir)
            self.assertEqual(manifest["model_id"], "gemini-3.1-flash-lite")
            self.assertEqual(manifest["requests"]["000001"]["sample_id"], "s1")

    def test_parse_batch_output_stores_valid_and_reports_failures(self) -> None:
        requests = {
            "000001": {
                "sample_id": "s1",
                "prompt_mode": "mixed",
                "prompt_signature": "sig",
                "fingerprint": "fp1",
                "task": "task1",
                "objects": ["image_0"],
            },
            "000002": {
                "sample_id": "s2",
                "prompt_mode": "mixed",
                "prompt_signature": "sig",
                "fingerprint": "fp2",
                "task": "task2",
                "objects": ["image_0"],
            },
            "000003": {
                "sample_id": "s3",
                "prompt_mode": "mixed",
                "prompt_signature": "sig",
                "fingerprint": "fp3",
                "task": "task3",
                "objects": ["image_0"],
            },
            "000004": {
                "sample_id": "s4",
                "prompt_mode": "mixed",
                "prompt_signature": "sig",
                "fingerprint": "fp4",
                "task": "task4",
                "objects": ["image_0"],
            },
        }
        error_line = json.dumps(
            {"custom_id": "000003", "response": {"status_code": 500}, "error": {"message": "boom"}}
        )
        lines = [
            _ok_line("000001", GOOD_CODE),
            _ok_line("000002", BAD_CODE),
            error_line,
            json.dumps({"custom_id": "999999", "response": {"status_code": 200}}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = FirstTurnStore(Path(tmp) / "first_turn.jsonl")
            report = parse_batch_output(
                lines, requests, store=store, parser=_StubParser(), model_id="m"
            )

            self.assertEqual(report.stored, 1)
            self.assertEqual(report.missing, ["000004"])
            failed_ids = {custom_id for custom_id, _ in report.failed}
            self.assertEqual(failed_ids, {"000002", "000003", "999999"})

            stored = store.get("s1", "mixed", "sig")
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.fingerprint, "fp1")
            self.assertEqual(stored.code, GOOD_CODE)
            self.assertEqual(stored.model_id, "m")
            self.assertIsNone(store.get("s2", "mixed", "sig"))

    def test_stored_entry_is_replayable_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "first_turn.jsonl"
            FirstTurnStore(path).put(
                FirstTurnEntry(
                    sample_id="s1",
                    prompt_mode="mixed",
                    prompt_signature="sig",
                    fingerprint="fp",
                    task="task",
                    objects=["image_0"],
                    thought="t",
                    code=GOOD_CODE,
                    raw_output=GOOD_CODE,
                    model_id="m",
                )
            )
            reloaded = FirstTurnStore(path).get("s1", "mixed", "sig")
            self.assertIsNotNone(reloaded)

    def test_collect_results_imports_local_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "batch"
            store_path = Path(tmp) / "first_turn.jsonl"
            write_batch_inputs(
                work_dir,
                [_request("000001", "s1")],
                benchmark="compbench",
                model_id="m",
                store_path=Path(tmp) / "manifest_store.jsonl",
            )
            output = Path(tmp) / "batch_output.jsonl"
            output.write_text(_ok_line("000001", GOOD_CODE) + "\n", encoding="utf-8")

            report = collect_results(
                work_dir, parser=_StubParser(), output_file=output, store_path=store_path
            )

            self.assertEqual(report.stored, 1)
            self.assertEqual(report.missing, [])
            self.assertIsNotNone(FirstTurnStore(store_path).get("s1", "mixed", "sig"))
            self.assertFalse((Path(tmp) / "manifest_store.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
