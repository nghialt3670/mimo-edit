#!/usr/bin/env python3
"""CompBench multi-object inference with Step1X-Edit, BAGEL, FLUX Kontext dev,
and Qwen-Image-Edit on Replicate.

Parallel, resumable inference with a bounded worker pool; metrics NOT computed.
Requires: pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

from PIL import Image, ImageOps

DATASET_ID = "BohanJia/CompBench"
TASKS = ("multi_object_add", "multi_object_remove")
MODELS = {
    # Community models: pin the published Replicate version for reproducibility.
    "step1x": {
        "slug": "zsxkib/step1x-edit",
        "version": "12b5a5a61e3419f792eb56cfc16eed046252740ebf5d470228f9b4cf2c861610",
    },
    "bagel": {
        "slug": "bytedance/bagel",
        "version": "7dd8def79e503990740db4704fa81af995d440fefe714958531d7044d2757c9c",
    },
    # Official endpoints do not accept pinning a community version in POST;
    # the returned prediction.version is recorded for auditability.
    "flux_kontext": {"slug": "black-forest-labs/flux-kontext-dev", "version": None},
    "qwen_image_edit": {"slug": "qwen/qwen-image-edit", "version": None},
}
OUTPUT_STATUS = ("succeeded", "failed", "canceled")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: dict) -> None:
    """Atomic metadata write, so interrupted runs can resume cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def safe_relative(image_path: str) -> Path:
    """Retain original CompBench filenames without allowing traversal."""
    path = PurePosixPath(str(image_path).replace("\\", "/"))
    if (
        not image_path
        or path.is_absolute()
        or any(p in ("..", "", ".") for p in path.parts)
    ):
        raise ValueError(f"Unsafe dataset image_path: {image_path!r}")
    return Path(*path.parts)


def sample_key(task: str, image_path: str) -> str:
    return hashlib.sha256((task + "\0" + image_path).encode("utf-8")).hexdigest()[:16]


def seed_for(task: str, image_path: str, base_seed: int) -> int:
    # Same sample gets the same seed across models, independent of iteration order.
    token = hashlib.sha256(f"{task}\0{image_path}".encode()).digest()
    return (base_seed + int.from_bytes(token[:4], "big")) % (2**31 - 1)


def save_image(image: Image.Image, dest: Path, *, mask: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(image)
    if mask:
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    # Saving temp files as explicit PNG avoids format detection on '.tmp'.
    tmp = dest.with_name(dest.name + ".tmp")
    image.save(tmp, format="PNG")
    tmp.replace(dest)


def image_to_pil(obj: Any) -> Image.Image:
    """Accept both decoded PIL and HF streaming decode(False) bytes/path."""
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            with Image.open(io.BytesIO(obj["bytes"])) as im:
                return im.copy()
        if obj.get("path") is not None:
            with Image.open(obj["path"]) as im:
                return im.copy()
    raise TypeError(f"Unsupported HF image type: {type(obj)}")


def exists_nonempty(path: Path) -> bool:
    """Fast check for cached reference inputs; generated outputs remain verified."""
    return path.is_file() and path.stat().st_size > 0


def png_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (OSError, ValueError):
        return False


def choose_samples(ds: Any, *, max_samples: int | None, max_per_task: int | None):
    """Filter before decoding images when using streaming datasets."""
    accepted = Counter()
    chosen = 0
    for row in ds:
        task = str(row["task"])
        if task not in TASKS:
            continue
        if max_per_task is not None and accepted[task] >= max_per_task:
            continue
        accepted[task] += 1
        chosen += 1
        yield row
        if max_samples is not None and chosen >= max_samples:
            break
        if max_per_task is not None and all(accepted[t] >= max_per_task for t in TASKS):
            break


def prepare_samples(
    root: Path,
    *,
    max_samples: int | None = None,
    max_per_task: int | None = None,
    streaming: bool = True,
    dataset: Any = None,
) -> list[dict]:
    """Download ONLY target samples' reference images and metadata.

    `dataset` is injectable for offline tests. The `train` split is deliberate.
    """
    if dataset is None:
        from datasets import load_dataset

        dataset = load_dataset(DATASET_ID, split="train", streaming=streaming)
        if streaming and hasattr(dataset, "decode"):
            # Avoid decoding three images on every irrelevant row in train.
            dataset = dataset.decode(False)

    ref_root = root / "tasks" / "multi_turn_editing"
    refs = []
    unique = set()
    for row in choose_samples(
        dataset, max_samples=max_samples, max_per_task=max_per_task
    ):
        task = str(row["task"])
        image_path = str(row["image_path"])
        rel = safe_relative(image_path)
        key = sample_key(task, image_path)
        if (task, image_path) in unique:
            raise ValueError(f"Duplicate CompBench sample {task}: {image_path}")
        unique.add((task, image_path))
        caption = row.get("caption") or ""
        parts = [p.strip() for p in str(caption).split("|", 1)]
        parts += [""] * (2 - len(parts))
        instruction = str(row.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"Missing editing instruction: {task}/{image_path}")
        src_dir = ref_root / task
        original = src_dir / "ori_images" / rel
        gt = src_dir / "edited_images" / rel
        mask = src_dir / "masks" / rel
        # Always write PNG reference data using original stem; source paths in
        # current dataset are .png (validate rather than silently rename).
        if rel.suffix.lower() != ".png":
            raise ValueError(f"Expected .png image_path for evaluator: {image_path}")
        for field, target, as_mask in (
            ("input_image", original, False),
            ("edited_image", gt, False),
            ("mask", mask, True),
        ):
            if not exists_nonempty(target):
                pil = row.get(field)
                if pil is None:
                    if field == "mask":
                        raise ValueError(
                            f"Missing required multi-object mask: {task}/{image_path}"
                        )
                    raise ValueError(f"Missing {field} for {task}/{image_path}")
                save_image(image_to_pil(pil), target, mask=as_mask)

        refs.append(
            {
                "id": key,
                "task": task,
                "image_path": image_path,
                "instruction": instruction,
                "caption": str(caption),
                "caption1": parts[0],
                "caption2": parts[1],
                "input_image": str(original.relative_to(root)),
                "edited_image": str(gt.relative_to(root)),
                "mask": str(mask.relative_to(root)),
            }
        )

    if not refs:
        raise RuntimeError("No multi-object samples found in CompBench train split")

    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.jsonl"
    selection_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in refs),
        encoding="utf-8",
    )
    # Compatible with CompBench eval_all.py metadata paths and caption keys.
    for task in TASKS:
        rows = [r for r in refs if r["task"] == task]
        target = ref_root / (task + ".jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        field = "add_ins" if task.endswith("add") else "remove_ins"
        with target.open("w", encoding="utf-8") as f:
            for r in rows:
                meta = {
                    "image_name": r["image_path"],
                    field: r["instruction"],
                    "caption1": r["caption1"],
                    "caption2": r["caption2"],
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return refs


def cached_or_prepare_samples(
    root: Path,
    *,
    max_samples: int | None,
    max_per_task: int | None,
    streaming: bool,
    refresh: bool = False,
    dataset: Any = None,
) -> list[dict]:
    selection = root / "selection.jsonl"
    settings_path = root / "selection_config.json"
    selection_args = {
        "dataset": DATASET_ID,
        "split": "train",
        "max_samples": max_samples,
        "max_per_task": max_per_task,
        "tasks": list(TASKS),
    }
    # Dataset argument is injected by offline tests; do not consume it when
    # a matching cached selection is available.
    if (
        not refresh
        and selection.is_file()
        and read_json(settings_path) == selection_args
    ):
        refs = [
            json.loads(line)
            for line in selection.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if refs and all(
            all(
                exists_nonempty(root / r[k])
                for k in ("input_image", "edited_image", "mask")
            )
            for r in refs
        ):
            print(
                f"Loaded {len(refs)} cached selected samples; no dataset scan needed.",
                flush=True,
            )
            return refs
        print("Cached dataset references incomplete; rebuilding selection.", flush=True)
    refs = prepare_samples(
        root,
        max_samples=max_samples,
        max_per_task=max_per_task,
        streaming=streaming,
        dataset=dataset,
    )
    write_json(settings_path, selection_args)
    return refs


class SubmissionLimiter:
    """Single shared prediction POST gate. Adjusts globally on 429 responses."""

    def __init__(self, min_interval: float):
        self.interval = min_interval
        self.lock = Lock()
        self.next_at = 0.0
        self.low_credit_mode = False

    def wait(self):
        # Re-check after waking: another worker may have raised the rate limit
        # while we slept. This prevents previously booked slots bypassing it.
        while True:
            with self.lock:
                now = time.monotonic()
                delay = self.next_at - now
                if delay <= 0:
                    self.next_at = now + self.interval
                    return
            time.sleep(max(0.01, delay))

    def throttle(self, reset_seconds: float = 10.0):
        """After 429, enforce <= 6 creates/min (11s gap), globally."""
        with self.lock:
            first = not self.low_credit_mode
            self.low_credit_mode = True
            self.interval = max(self.interval, 11.0)
            self.next_at = max(
                self.next_at, time.monotonic() + max(11.0, reset_seconds + 1.0)
            )
        if first:
            print(
                "Replicate 429: switching ALL workers to <= 6 creates/min (11s spacing). "
                "Check available credit and API-token account.",
                flush=True,
            )


def is_rejected_429(exc: Exception | str) -> bool:
    """Recognize a definite HTTP 429, including replicate-python's text format."""
    for obj in (exc, getattr(exc, "response", None)):
        if obj is None:
            continue
        for attr in ("status_code", "status"):
            code = getattr(obj, attr, None)
            if str(code) == "429":
                return True
    message = str(exc)
    return bool(
        re.search(
            r"\b(?:status|status_code)\s*[:=]\s*429\b", message, flags=re.IGNORECASE
        )
        or re.search(r"\b429\s+Too Many Requests\b", message, flags=re.IGNORECASE)
    )


def rate_limit_reset_seconds(exc: Exception) -> float:
    """Parse either Retry-After or '~Ns' from Replicate's rejection detail."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("Retry-After", headers.get("retry-after", ""))
    try:
        return min(120.0, max(1.0, float(raw)))
    except (TypeError, ValueError):
        match = re.search(r"resets?\s+in\s+~?\s*(\d+(?:\.\d+)?)s", str(exc), re.I)
        return min(120.0, max(1.0, float(match.group(1)))) if match else 10.0


def model_input(
    model: str,
    prompt: str,
    input_file: Any,
    seed: int,
    *,
    bagel_thinking: bool = False,
    fast: bool = False,
    step1x_size_level: int = 512,
) -> dict:
    """Use each endpoint's actual input argument names, not one shared schema."""
    if model == "step1x":
        return {
            "image": input_file,
            "prompt": prompt,
            "seed": seed,
            "size_level": step1x_size_level,
            "output_format": "png",
        }
    if model == "bagel":
        return {
            "task": "image-editing",
            "image": input_file,
            "prompt": prompt,
            "seed": seed,
            "enable_thinking": bagel_thinking,
            "num_inference_steps": 50,
            "cfg_img_scale": 2,
            "cfg_text_scale": 4,
            "cfg_renorm_min": 1,
            "cfg_renorm_type": "text_channel",
            "timestep_shift": 3,
            "output_format": "png",
        }
    if model == "flux_kontext":
        return {
            "input_image": input_file,
            "prompt": prompt,
            "seed": seed,
            "go_fast": fast,
            "aspect_ratio": "match_input_image",
            "guidance": 2.5,
            "num_inference_steps": 28,
            "output_format": "png",
        }
    if model == "qwen_image_edit":
        return {
            "image": input_file,
            "prompt": prompt,
            "seed": seed,
            "go_fast": fast,
            "aspect_ratio": "match_input_image",
            "output_format": "png",
        }
    raise KeyError(f"Unknown model: {model}")


def output_image_url(output: Any) -> str:
    """Supports URI, one-element list, BAGEL's {'image': URI, 'text':...}."""
    if isinstance(output, str) and output.startswith(("https://", "http://")):
        return output
    if isinstance(output, (list, tuple)) and output:
        return output_image_url(output[0])
    if isinstance(output, dict):
        for k in ("image", "images", "output"):
            if output.get(k):
                return output_image_url(output[k])
    # Replicate FileOutput may expose .url
    url = getattr(output, "url", None)
    if callable(url):
        url = url()
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        return url
    raise ValueError(f"Cannot find image URL in prediction output: {str(output)[:500]}")


def download_result_image(url: str, dest: Path) -> tuple[int, int]:
    """Download promptly (Replicate output URLs are not permanent), verify/save PNG."""
    import requests

    response = requests.get(url, timeout=(20, 120))
    response.raise_for_status()
    if len(response.content) > 100 * 1024 * 1024:
        raise ValueError("Output image unexpectedly exceeds 100MB")
    with Image.open(io.BytesIO(response.content)) as raw:
        im = raw.copy()
        dims = im.size
        save_image(im, dest)
    return dims


def create_prediction(client: Any, model: str, kwargs: dict):
    cfg = MODELS[model]
    if cfg["version"]:
        return client.predictions.create(
            version=f"{cfg['slug']}:{cfg['version']}", input=kwargs
        )
    return client.predictions.create(model=cfg["slug"], input=kwargs)


def create_with_rate_limit_retry(
    client: Any, model: str, kwargs: dict, limiter: SubmissionLimiter | None = None
):
    """Only 429 is safe to retry: Replicate explicitly rejected that POST.

    Network timeouts or other POST errors must NOT be automatically retried:
    the remote request could already have started and incurred a charge.
    """
    for attempt in range(5):
        if limiter is not None:
            limiter.wait()
        try:
            return create_prediction(client, model, kwargs)
        except Exception as exc:
            if not is_rejected_429(exc):
                raise  # A non-429 POST error may have produced a billed job.
            delay = rate_limit_reset_seconds(exc)
            if limiter is not None:
                limiter.throttle(delay)
            if attempt == 4:
                raise  # Explicitly rejected; safe to resume this job later.
            if limiter is None:
                time.sleep(max(11.0, delay + 1.0))
            print(
                f"Replicate 429: retrying after shared limiter permits it "
                f"(attempt {attempt + 2}/5)",
                flush=True,
            )
    raise AssertionError("unreachable")


def predict_until_done(
    client: Any, pred_id: str, *, poll_seconds: float, max_wait: float
):
    start = time.monotonic()
    failures = 0
    while True:
        try:
            p = client.predictions.get(pred_id)
            failures = 0
        except Exception:
            failures += 1
            if failures >= 5:
                raise
            time.sleep(min(3.0 * failures, 15.0))
            continue
        if p.status in OUTPUT_STATUS:
            return p
        if time.monotonic() - start > max_wait:
            raise TimeoutError(
                f"Prediction {pred_id} still {p.status} after {max_wait}s. "
                "It was NOT canceled; rerun to retrieve it without re-submitting."
            )
        time.sleep(poll_seconds)


def result_paths(root: Path, model: str, ref: dict) -> tuple[Path, Path]:
    rel = safe_relative(ref["image_path"])
    task = ref["task"]
    # Exact CompBench evaluator convention:
    # editing_results/<model>/<model>_multi/<task>/<image_name>
    png = root / "editing_results" / model / f"{model}_multi" / task / rel
    record = root / "records" / model / task / rel.with_suffix(".json")
    return png, record


def compact_input(settings: dict) -> dict:
    return {
        k: ("[local input image]" if hasattr(v, "read") else v)
        for k, v in settings.items()
    }


def process_one(
    root: Path,
    ref: dict,
    model: str,
    *,
    client: Any,
    base_seed: int,
    poll_seconds: float,
    max_wait: float,
    bagel_thinking: bool,
    fast: bool,
    step1x_size_level: int,
    retry_failed: bool = False,
    limiter: SubmissionLimiter | None = None,
) -> str:
    png, record_file = result_paths(root, model, ref)
    record = read_json(record_file)
    seed = seed_for(ref["task"], ref["image_path"], base_seed)
    expected_inputs = model_input(
        model,
        ref["instruction"],
        "[local input image]",
        seed,
        bagel_thinking=bagel_thinking,
        fast=fast,
        step1x_size_level=step1x_size_level,
    )
    if record:
        if (
            record.get("settings") != expected_inputs
            or record.get("prompt") != ref["instruction"]
            or record.get("requested_version") != MODELS[model]["version"]
        ):
            raise RuntimeError(
                f"Existing prediction settings differ: {model} / {ref['image_path']}. "
                "Use a NEW --output-dir for a different experimental configuration."
            )
    if png_is_valid(png) and record.get("status") == "succeeded":
        return "skipped"
    if (
        record.get("status") in ("failed", "canceled", "create_error")
        and not retry_failed
    ):
        return "prior_failure"
    # Repair existing v2 429 records. Replicate rejected these requests; there
    # is no billed prediction to duplicate. Real ambiguous POSTs remain frozen.
    if record.get("status") in ("creating", "create_uncertain") and not record.get(
        "prediction_id"
    ):
        if record.get("status") == "create_uncertain" and is_rejected_429(
            record.get("error", "")
        ):
            record["status"] = "rate_limited"
            write_json(record_file, record)
        else:
            return "create_uncertain"

    now = utc_now()
    source = root / ref["input_image"]
    pred_id = record.get("prediction_id")
    if not pred_id or (
        record.get("status") in ("failed", "canceled", "create_error") and retry_failed
    ):
        with source.open("rb") as src:
            submission_started = time.monotonic()
            inputs = model_input(
                model,
                ref["instruction"],
                src,
                seed,
                bagel_thinking=bagel_thinking,
                fast=fast,
                step1x_size_level=step1x_size_level,
            )
            new_record = {
                "sample_id": ref["id"],
                "task": ref["task"],
                "image_path": ref["image_path"],
                "model": model,
                "model_slug": MODELS[model]["slug"],
                "requested_version": MODELS[model]["version"],
                "prompt": ref["instruction"],
                "seed": seed,
                "settings": compact_input(inputs),
                "submitted_at": now,
                "status": "creating",
                "prediction_id": None,
                "output_path": str(png.relative_to(root)),
            }
            write_json(record_file, new_record)
            try:
                pred = create_with_rate_limit_retry(client, model, inputs, limiter)
            except Exception as exc:
                # HTTP 429 definitively rejects creation: it is safe to retry
                # on a later run. Other POST errors may have incurred charges.
                rejected = is_rejected_429(exc)
                new_record.update(
                    {
                        "status": "rate_limited" if rejected else "create_uncertain",
                        "error": str(exc),
                        "updated_at": utc_now(),
                    }
                )
                write_json(record_file, new_record)
                if rejected:
                    return "rate_limited"
                raise
        pred_id = pred.id
        new_record.update(
            {
                "status": pred.status,
                "prediction_id": pred_id,
                "submission_wall_seconds": round(
                    time.monotonic() - submission_started, 2
                ),
                "resolved_version": getattr(pred, "version", None),
                "updated_at": utc_now(),
            }
        )
        write_json(record_file, new_record)
        record = new_record

    try:
        retrieval_started = time.monotonic()
        # A timed-out/interrupted run resumes by fetching the SAME id.
        pred = predict_until_done(
            client, pred_id, poll_seconds=poll_seconds, max_wait=max_wait
        )
        record.update(
            {
                "status": pred.status,
                "resolved_version": getattr(pred, "version", None),
                "metrics": getattr(pred, "metrics", None),
                "prediction_error": getattr(pred, "error", None),
                "updated_at": utc_now(),
            }
        )
        if pred.status != "succeeded":
            record["error"] = str(getattr(pred, "error", "Model failed"))
            write_json(record_file, record)
            return "failed"
        if pred.output is None:
            raise RuntimeError(
                "Replicate output has expired (usually ~1 hour after completion); "
                "cannot recover this image without a new paid prediction."
            )
        url = output_image_url(pred.output)
        record["output_url"] = url
        # Always record URL first: if downloading fails, resume without a new call.
        write_json(record_file, record)
        w, h = download_result_image(url, png)
        record.update(
            {
                "image_size": [w, h],
                "status": "succeeded",
                "saved_at": utc_now(),
                "retrieve_save_wall_seconds": round(
                    time.monotonic() - retrieval_started, 2
                ),
            }
        )
        write_json(record_file, record)
        return "succeeded"
    except Exception as exc:
        record.update({"last_retrieval_error": repr(exc), "updated_at": utc_now()})
        if record.get("status") not in ("succeeded", "failed", "canceled"):
            record["status"] = "pending_retrieval"
        write_json(record_file, record)
        raise


def write_summary(
    root: Path, refs: list[dict], models: list[str], *, base_seed: int = 42
) -> dict:
    rows = []
    counts = Counter()
    for ref in refs:
        for model in models:
            png, record_path = result_paths(root, model, ref)
            rec = read_json(record_path)
            state = rec.get("status", "not_started")
            # A succeeded prediction is not an evaluated result until saved locally.
            image_present = png_is_valid(png)
            if state == "succeeded" and not image_present:
                state = "missing_local_image"
            counts[(model, state)] += 1
            rows.append(
                {
                    "model": model,
                    "task": ref["task"],
                    "image_path": ref["image_path"],
                    "sample_id": ref["id"],
                    "instruction": ref["instruction"],
                    "caption1": ref["caption1"],
                    "caption2": ref["caption2"],
                    "seed": rec.get(
                        "seed", seed_for(ref["task"], ref["image_path"], base_seed)
                    ),
                    "status": state,
                    "prediction_id": rec.get("prediction_id", ""),
                    "resolved_version": rec.get("resolved_version", ""),
                    "output_path": str(png.relative_to(root)) if image_present else "",
                }
            )
    fields = list(rows[0].keys()) if rows else []
    with (root / "inference_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        model: dict(
            Counter(
                {state: count for (m, state), count in counts.items() if m == model}
            )
        )
        for model in models
    }
    write_json(
        root / "summary.json",
        {"updated_at": utc_now(), "total_samples": len(refs), "models": summary},
    )
    return summary


def preflight_existing(
    root: Path,
    refs: list[dict],
    models: list[str],
    *,
    seed: int,
    bagel_thinking: bool,
    fast: bool,
    step1x_size_level: int,
):
    """Refuse mixed configurations BEFORE scheduling paid work."""
    for ref in refs:
        for model in models:
            _, rec_path = result_paths(root, model, ref)
            if not rec_path.is_file():
                continue
            rec = read_json(rec_path)
            expected = model_input(
                model,
                ref["instruction"],
                "[local input image]",
                seed_for(ref["task"], ref["image_path"], seed),
                bagel_thinking=bagel_thinking,
                fast=fast,
                step1x_size_level=step1x_size_level,
            )
            if (
                rec.get("settings") != expected
                or rec.get("prompt") != ref["instruction"]
                or rec.get("requested_version") != MODELS[model]["version"]
            ):
                raise RuntimeError(
                    f"Existing settings differ for {model}: {ref['image_path']}. "
                    "Use a new --output-dir to avoid mixed experiment results."
                )


def run_benchmark(args: argparse.Namespace, *, dataset: Any = None, client: Any = None):
    root = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run and client is None and not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("Set REPLICATE_API_TOKEN, or use --dry-run (no API calls).")
    if not args.dry_run and client is None:
        import replicate

        client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    refs = cached_or_prepare_samples(
        root,
        max_samples=args.max_samples,
        max_per_task=args.max_per_task,
        streaming=args.streaming,
        refresh=getattr(args, "refresh_dataset", False),
        dataset=dataset,
    )
    print(
        f"Prepared {len(refs)} samples: {dict(Counter(r['task'] for r in refs))}",
        flush=True,
    )
    config = {
        "dataset": DATASET_ID,
        "split": "train",
        "tasks": list(TASKS),
        "models": args.models,
        "model_specs": {m: MODELS[m] for m in args.models},
        "base_seed": args.seed,
        "bagel_thinking": args.bagel_thinking,
        "fast": args.fast,
        "step1x_size_level": args.step1x_size_level,
        "max_samples": args.max_samples,
        "max_per_task": args.max_per_task,
        "dataset_streaming": args.streaming,
        "workers": args.workers,
        "submit_interval": args.submit_interval,
        "updated_at": utc_now(),
    }
    if args.dry_run:
        write_json(root / "run_config.json", config)
        print("DRY RUN: no Replicate predictions created; images/GT/masks saved.")
        print(f"Output: {root}")
        return write_summary(root, refs, args.models, base_seed=args.seed)

    preflight_existing(
        root,
        refs,
        args.models,
        seed=args.seed,
        bagel_thinking=args.bagel_thinking,
        fast=args.fast,
        step1x_size_level=args.step1x_size_level,
    )
    write_json(root / "run_config.json", config)
    total_jobs = len(refs) * len(args.models)
    limiter = SubmissionLimiter(args.submit_interval)
    counters = Counter()

    def work(ref: dict, model: str) -> str:
        return process_one(
            root,
            ref,
            model,
            client=client,
            base_seed=args.seed,
            poll_seconds=args.poll_seconds,
            max_wait=args.max_wait,
            bagel_thinking=args.bagel_thinking,
            fast=args.fast,
            step1x_size_level=args.step1x_size_level,
            retry_failed=args.retry_failed,
            limiter=limiter,
        )

    def jobs():
        for ref in refs:
            for model in args.models:
                yield ref, model

    # Bounded in-flight jobs prevent a huge full-dataset future queue.
    pending = {}
    iterator = iter(jobs())
    done = 0
    start = time.monotonic()

    def enqueue(pool):
        try:
            ref, model = next(iterator)
        except StopIteration:
            return False
        future = pool.submit(work, ref, model)
        pending[future] = (ref, model)
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in range(min(args.workers * 2, total_jobs)):
            enqueue(pool)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                ref, model = pending.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = "error"
                    print(
                        f"[{done + 1}/{total_jobs}] {model} {ref['task']}/{ref['image_path']}: "
                        f"ERROR: {exc}",
                        flush=True,
                    )
                    if args.stop_on_error:
                        # No new requests after the first error; running jobs
                        # already submitted remain in flight and will finish.
                        for other in pending:
                            other.cancel()
                        write_summary(root, refs, args.models, base_seed=args.seed)
                        raise
                else:
                    print(
                        f"[{done + 1}/{total_jobs}] {model} {ref['task']}/{ref['image_path']}: "
                        f"{status}",
                        flush=True,
                    )
                done += 1
                counters[(model, status)] += 1
                if args.sleep_seconds > 0 and status == "succeeded":
                    time.sleep(args.sleep_seconds)
                enqueue(pool)
            if done % 50 == 0 or done == total_jobs:
                elapsed = time.monotonic() - start
                print(
                    f"Progress: {done}/{total_jobs} ({elapsed:.1f}s elapsed).",
                    flush=True,
                )
            # Manifest scans every output on disk; checkpoint only a few times.
            if done == total_jobs or done % max(100, total_jobs // 4) == 0:
                write_summary(root, refs, args.models, base_seed=args.seed)

    summary = write_summary(root, refs, args.models, base_seed=args.seed)
    print(f"Done. Summary: {json.dumps(summary, indent=2)}")
    print(f"Results in: {root}")
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 4 image editors via Replicate on CompBench multi-object subset"
    )
    parser.add_argument(
        "--output-dir",
        default=".out/evaluation/compbench/replicate",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit overall selected CompBench samples (default all).",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="Balanced smoke test: e.g. 1 add + 1 remove.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream HF train data to avoid downloading entire split.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="No Replicate calls or cost."
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Resubmit only previously failed/canceled predictions (billed again).",
    )
    parser.add_argument("--bagel-thinking", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable go_fast for Kontext and Qwen (quality may change).",
    )
    parser.add_argument("--step1x-size-level", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent Replicate jobs (default 4).",
    )
    parser.add_argument(
        "--submit-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between prediction POSTs, across all workers. "
        "Use 11 for accounts with < $5 balance; 429 also triggers adaptive spacing.",
    )
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="Rebuild dataset selection even if cached on disk.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-wait",
        type=float,
        default=2400.0,
        help="Max polling time per prediction; it is NOT canceled on timeout.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be >= 1")
    if args.max_per_task is not None and args.max_per_task <= 0:
        raise SystemExit("--max-per-task must be >= 1")
    if args.step1x_size_level <= 0 or args.poll_seconds <= 0 or args.max_wait <= 0:
        raise SystemExit("Size level, poll seconds, and max wait must be positive")
    if args.workers < 1 or args.submit_interval < 0:
        raise SystemExit("--workers must be >= 1 and --submit-interval must be >= 0")
    run_benchmark(args)


if __name__ == "__main__":
    main()
