
#!/usr/bin/env python3
"""
Submit an OpenAI-format JSONL batch to Gemini, monitor progress,
and download the output and error files.

Install:
    python -m pip install -U openai google-genai

Set API key:
    export GEMINI_API_KEY="your-api-key"

Run:
    python src/scripts/submit_batch.py requests.jsonl

Optional:
    python src/scripts/submit_batch.py requests.jsonl --no-wait
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI

from framework.utils.project import load_project_environment

load_project_environment()

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def convert_for_gemini_batch(input_path: Path) -> Path:
    """Replace unsupported json_schema response_format with json_object,
    and move the schema requirements into the system prompt.
    """
    output_path = input_path.with_name(
        input_path.stem + "_gemini_compatible.jsonl"
    )

    converted = 0

    with input_path.open("r", encoding="utf-8") as src, \
         output_path.open("w", encoding="utf-8") as dst:

        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            body = item["body"]
            response_format = body.get("response_format", {})

            if response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema", {})
                schema = json_schema.get("schema")

                if schema:
                    schema_instruction = (
                        "\n\nReturn ONLY a valid JSON object matching "
                        "this JSON Schema. Do not use Markdown fences.\n"
                        "JSON Schema:\n"
                        + json.dumps(schema, ensure_ascii=False)
                    )

                    messages = body.setdefault("messages", [])
                    system_message = next(
                        (
                            message for message in messages
                            if message.get("role") == "system"
                        ),
                        None,
                    )

                    if system_message is None:
                        messages.insert(
                            0,
                            {
                                "role": "system",
                                "content": schema_instruction,
                            },
                        )
                    else:
                        content = system_message.get("content", "")

                        if isinstance(content, str):
                            system_message["content"] = (
                                content + schema_instruction
                            )
                        elif isinstance(content, list):
                            content.append({
                                "type": "text",
                                "text": schema_instruction,
                            })
                        else:
                            raise ValueError(
                                f"Unexpected system message format "
                                f"on line {line_number}"
                            )

                body["response_format"] = {"type": "json_object"}
                converted += 1

            dst.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {converted} structured-output requests.")
    print(f"Compatible input written to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit and monitor a Gemini batch job."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="requests.jsonl",
        help="Path to OpenAI-format JSONL input (default: requests.jsonl)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status checks (default: 30)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the batch and exit without polling for completion.",
    )
    parser.add_argument(
        "--display-name",
        default="mixed-prompt-batch",
        help="Display name for the uploaded file.",
    )
    return parser.parse_args()


def validate_jsonl(path: Path) -> int:
    """Validate the OpenAI Batch API JSONL structure."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"Line {line_number}: each JSONL line must be an object."
                )

            for field in ("custom_id", "method", "url", "body"):
                if field not in item:
                    raise ValueError(
                        f"Line {line_number}: missing required field "
                        f"{field!r}."
                    )

            if not isinstance(item["body"], dict):
                raise ValueError(
                    f"Line {line_number}: 'body' must be a JSON object."
                )

            if not isinstance(item["custom_id"], str):
                raise ValueError(
                    f"Line {line_number}: 'custom_id' must be a string."
                )

            if item["method"] != "POST":
                raise ValueError(
                    f"Line {line_number}: expected method 'POST', "
                    f"got {item['method']!r}."
                )

            if item["url"] != "/v1/chat/completions":
                raise ValueError(
                    f"Line {line_number}: unexpected endpoint "
                    f"{item['url']!r}."
                )

            count += 1

    if count == 0:
        raise ValueError("The input file contains no JSONL requests.")

    return count


def get_request_counts(batch: Any) -> str:
    """Format batch progress when request counts are available."""
    counts = getattr(batch, "request_counts", None)

    if counts is None:
        return "request counts unavailable"

    completed = getattr(counts, "completed", None)
    failed = getattr(counts, "failed", None)
    total = getattr(counts, "total", None)

    parts = []

    if completed is not None:
        parts.append(f"completed={completed}")
    if failed is not None:
        parts.append(f"failed={failed}")
    if total is not None:
        parts.append(f"total={total}")

    return ", ".join(parts) if parts else str(counts)


def download_file(
    google_client: genai.Client,
    file_id: str | None,
    destination: Path,
) -> None:
    """Download a Gemini file and save it locally."""
    if not file_id:
        print(f"No file available for {destination.name}.")
        return

    print(f"Downloading {file_id} -> {destination}")

    content = google_client.files.download(file=file_id)

    if isinstance(content, bytes):
        data = content
    elif isinstance(content, bytearray):
        data = bytes(content)
    elif isinstance(content, str):
        data = content.encode("utf-8")
    else:
        # Some SDK versions may return a file-like object.
        if hasattr(content, "read"):
            data = content.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
        else:
            data = bytes(content)

    destination.write_bytes(data)
    print(f"Saved {destination} ({len(data):,} bytes)")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()

    if args.poll_interval < 1:
        print("Error: --poll-interval must be at least 1 second.")
        return 2

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("MIXED_PROMPT_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        print(
            "Error: GEMINI_API_KEY, MIXED_PROMPT_API_KEY or GOOGLE_API_KEY is not set.\n"
            "macOS/Linux: export GEMINI_API_KEY='your-api-key'\n"
            "PowerShell:  $env:GEMINI_API_KEY='your-api-key'",
            file=sys.stderr,
        )
        return 2

    try:
        count = validate_jsonl(input_path)
        print(f"Validated {count} requests.")
        print(f"Input: {input_path}")

        # Google SDK handles file uploads and downloads.
        google_client = genai.Client(api_key=api_key)

        # OpenAI-compatible Gemini API handles batch operations.
        openai_client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
        )

        batch_input_path = convert_for_gemini_batch(input_path)

        print("\nUploading Gemini-compatible input file...")
        with batch_input_path.open("rb") as file:
            uploaded = google_client.files.upload(
                file=file,
                config=types.UploadFileConfig(
                    display_name="mixed-prompt-batch",
                    mime_type="jsonl",
                ),
            )

        if not uploaded.name:
            raise RuntimeError(
                "Upload succeeded but Google returned no file name."
            )

        print(f"Uploaded file: {uploaded.name}")

        # 2. Submit the batch job.
        # Use the Google file name as input_file_id.
        print("\nCreating Gemini batch job...")
        batch = openai_client.batches.create(
            input_file_id=uploaded.name,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        batch_id = batch.id
        print(f"Batch ID: {batch_id}")
        print(f"Initial status: {batch.status}")

        # Save the ID so the job can be checked later if this process exits.
        batch_id_path = Path("batch_id.txt").resolve()
        batch_id_path.write_text(batch_id + "\n", encoding="utf-8")
        print(f"Batch ID saved to: {batch_id_path}")

        if args.no_wait:
            print(
                "\nSubmitted successfully. You can check this batch later "
                f"using its ID: {batch_id}"
            )
            return 0

        # 3. Poll until the job reaches a terminal status.
        print("\nWaiting for batch completion...")
        while batch.status not in TERMINAL_STATUSES:
            print(
                f"Status: {batch.status} | "
                f"{get_request_counts(batch)}"
            )
            time.sleep(args.poll_interval)
            batch = openai_client.batches.retrieve(batch_id)

        print(f"\nFinal status: {batch.status}")
        print(f"Final counts: {get_request_counts(batch)}")

        # 4. Download output and error files, if present.
        output_path = Path("batch_output.jsonl").resolve()
        error_path = Path("batch_errors.jsonl").resolve()

        download_file(
            google_client,
            getattr(batch, "output_file_id", None),
            output_path,
        )
        download_file(
            google_client,
            getattr(batch, "error_file_id", None),
            error_path,
        )

        if batch.status == "completed":
            print("\nBatch finished. Check batch_output.jsonl.")
            return 0

        print(
            "\nThe batch did not complete successfully. "
            "Inspect batch_errors.jsonl and the final status above.",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "\nInterrupted. The remote batch may still be running. "
            "Check batch_id.txt before submitting again.",
            file=sys.stderr,
        )
        return 130

    except Exception as exc:
        print(
            f"\nError: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
