from __future__ import annotations

import importlib
import json
import math
import numbers
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from evaluation.common import HFImageLoader, read_jsonl
from evaluation.compbench_metrics import collect_saved_outputs
from evaluation.compbench_viescore import (
    MULTI_OBJECT_TASKS,
    collect_replicate_saved_outputs,
)

EDITSCORE_NAMES = (
    "prompt_following",
    "consistency",
    "perceptual_quality",
    "overall",
)


def _enable_editscore_mps(backbone: str) -> bool:
    """Patch the official Qwen adapters to load and place inputs on Apple MPS."""
    import torch

    mps_available = torch.backends.mps.is_available()
    requested_device = os.environ.get("EDITSCORE_DEVICE", "auto").lower()
    if requested_device not in {"auto", "mps"}:
        raise ValueError("EDITSCORE_DEVICE must be 'auto' or 'mps'.")
    if requested_device == "mps" and not mps_available:
        raise EnvironmentError(
            "EDITSCORE_DEVICE=mps was requested, but this PyTorch build has no "
            "available MPS device."
        )
    if not mps_available or (requested_device == "auto" and torch.cuda.is_available()):
        return False

    adapter_names = {
        "qwen3vl": ("editscore.mllm_tools.qwen3vl", "Qwen3VL"),
        "qwen25vl": ("editscore.mllm_tools.qwen25vl", "Qwen25VL"),
    }
    try:
        module_name, class_name = adapter_names[backbone]
    except KeyError as exc:
        raise ValueError(
            f"EditScore MPS support is available for qwen3vl and qwen25vl, "
            f"not {backbone!r}."
        ) from exc

    module = importlib.import_module(module_name)
    upstream_class = getattr(module, class_name)
    if getattr(upstream_class, "_mps_compatible", False):
        return True

    model_class = getattr(
        module,
        (
            "Qwen3VLForConditionalGeneration"
            if backbone == "qwen3vl"
            else "Qwen2_5_VLForConditionalGeneration"
        ),
    )
    processor_class = module.AutoProcessor
    peft_model = module.PeftModel

    class MPSQwenAdapter(upstream_class):
        _mps_compatible = True

        def __init__(
            self,
            vlm_model: str,
            temperature: float = 0.7,
            seed: int | None = None,
            lora_path: str | None = None,
        ) -> None:

            self.model = model_class.from_pretrained(
                vlm_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            if lora_path:
                self.model = peft_model.from_pretrained(
                    self.model,
                    lora_path,
                )

            self.model = self.model.to("mps").eval()

            self.processor = processor_class.from_pretrained(vlm_model)

            self.temperature = temperature
            self.seed = seed

        def prepare_input(
            self,
            images,
            text_prompt: str = "",
        ):
            if not isinstance(images, list):
                images = [images]

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image} for image in images]
                    + [{"type": "text", "text": text_prompt}],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            return inputs.to(self.model.device)

    MPSQwenAdapter.__name__ = f"MPS{class_name}"
    setattr(module, class_name, MPSQwenAdapter)
    print(f"Using EditScore {backbone} on Apple MPS (float16).")
    return True


def create_editscore_scorer(
    *,
    backbone: str | None = None,
    model_name_or_path: str | None = None,
    lora_path: str | None = None,
    score_range: int = 25,
    num_pass: int = 1,
) -> Any:
    """Create the official local EditScore reward model."""
    backbone = backbone or os.environ.get("EDITSCORE_BACKBONE", "qwen3vl")
    _enable_editscore_mps(backbone)
    try:
        from editscore import EditScore
    except ImportError as exc:
        raise ImportError(
            "CompBench EditScore requires the official editscore package. "
            "Install it in the active environment with `pip install -U editscore`."
        ) from exc

    model_name_or_path = model_name_or_path or os.environ.get(
        "EDITSCORE_MODEL", "Qwen/Qwen3-VL-4B-Instruct"
    )
    lora_path = lora_path or os.environ.get(
        "EDITSCORE_LORA", "EditScore/EditScore-Qwen3-VL-4B-Instruct"
    )
    return EditScore(
        backbone=backbone,
        model_name_or_path=model_name_or_path,
        lora_path=lora_path,
        score_range=score_range,
        num_pass=num_pass,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_cached_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        cache_key = row.get("cache_key")
        if isinstance(cache_key, str):
            cached[cache_key] = row
    return cached


def _cache_key(
    *,
    model: str | None,
    sample_id: str,
    instruction: str,
    source_identity: str,
    source_path: Path | None,
    output_path: Path,
) -> str:
    output_stat = output_path.stat()
    source_stat = source_path.stat() if source_path is not None else None
    payload = {
        "model": model,
        "sample_id": sample_id,
        "instruction": instruction,
        "source_identity": source_identity,
        "source_path": str(source_path) if source_path is not None else None,
        "source_size_bytes": source_stat.st_size if source_stat else None,
        "source_mtime_ns": source_stat.st_mtime_ns if source_stat else None,
        "output_path": str(output_path),
        "output_size_bytes": output_stat.st_size,
        "output_mtime_ns": output_stat.st_mtime_ns,
        "backbone": os.environ.get("EDITSCORE_BACKBONE", "qwen3vl"),
        "model_name_or_path": os.environ.get(
            "EDITSCORE_MODEL", "Qwen/Qwen3-VL-4B-Instruct"
        ),
        "lora_path": os.environ.get(
            "EDITSCORE_LORA", "EditScore/EditScore-Qwen3-VL-4B-Instruct"
        ),
        "num_pass": os.environ.get("EDITSCORE_NUM_PASS", "1"),
        "score_range": os.environ.get("EDITSCORE_SCORE_RANGE", "25"),
        "device": os.environ.get("EDITSCORE_DEVICE", "auto"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _resize_output(source: Image.Image, output: Image.Image) -> Image.Image:
    if output.size == source.size:
        return output.convert("RGB")
    return output.convert("RGB").resize(source.size, Image.Resampling.BICUBIC)


def _score(
    scorer: Any,
    source: Image.Image,
    output_path: Path,
    instruction: str,
) -> dict[str, float]:
    output = _resize_output(source, _load_rgb_image(output_path))
    result = scorer.evaluate([source.convert("RGB"), output], instruction)
    if not isinstance(result, dict):
        raise TypeError(f"Expected an EditScore result dictionary, got {result!r}")
    scores: dict[str, float] = {}
    for name in EDITSCORE_NAMES:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"EditScore result is missing numeric {name!r}: {result!r}"
            )
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"EditScore returned non-finite {name}: {score!r}")
        scores[name] = score
    return scores


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(group: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            **{
                name: fmean(float(row[name]) for row in group)
                for name in EDITSCORE_NAMES
            },
            "count": len(group),
        }

    summary: dict[str, Any] = {}
    if rows:
        summary["overall"] = group_summary(rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    summary["by_task"] = {
        task: group_summary(group) for task, group in sorted(by_task.items())
    }
    if any("model" in row for row in rows):
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_model[str(row["model"])].append(row)
        summary["by_model"] = {
            model: {
                "overall": group_summary(group),
                "by_task": {
                    task: group_summary([row for row in group if row["task"] == task])
                    for task in sorted({str(row["task"]) for row in group})
                },
            }
            for model, group in sorted(by_model.items())
        }
    return summary


def _save_results(
    rows: list[dict[str, Any]], output_dir: Path, metric_stem: str
) -> dict[str, Any]:
    metrics_dir = output_dir / "metrics"
    metric_path = metrics_dir / f"{metric_stem}.jsonl"
    _write_jsonl(metric_path, rows)
    summary = _summarize(rows)
    summary_path = metrics_dir / f"{metric_stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Per-sample EditScore: {metric_path}")
    print(f"EditScore summary: {summary_path}")
    return summary


def evaluate_saved_compbench_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    image_loader: HFImageLoader,
    prompt_mode: str,
    scorer: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score saved CompBench multi-object outputs with EditScore."""
    if prompt_mode not in ("text", "mixed"):
        raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")
    output_dir = output_dir.expanduser().resolve()
    saved_outputs = collect_saved_outputs(
        records,
        output_dir,
        prompt_mode=prompt_mode,
        task_filter=MULTI_OBJECT_TASKS,
    )
    if scorer is None:
        scorer = create_editscore_scorer(
            num_pass=int(os.environ.get("EDITSCORE_NUM_PASS", "1")),
            score_range=int(os.environ.get("EDITSCORE_SCORE_RANGE", "25")),
        )
    metric_path = output_dir / "metrics" / "editscore.jsonl"
    cached = _read_cached_rows(metric_path)
    rows: list[dict[str, Any]] = []
    for saved in tqdm(saved_outputs, desc="CompBench EditScore", unit="sample"):
        index = saved["record_position"]
        record = records[index]
        metadata = record.get("metadata") or {}
        task = metadata.get("task")
        if task not in MULTI_OBJECT_TASKS:
            raise ValueError(
                f"Unexpected CompBench multi-object task for "
                f"{record.get('sample_id')}: {task!r}"
            )
        instruction = record.get("original_instruction")
        sample_id = record.get("sample_id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Missing original_instruction for {sample_id}")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("CompBench annotation is missing sample_id")

        output_path: Path = saved["output_path"]
        source_ref = record.get("source_ref")
        source_reference = image_loader.parse_reference(source_ref)
        if source_reference is None:
            raise ValueError(f"Missing source_ref for {sample_id}")
        dataset_id, split, row_index, image_field = source_reference
        cache_key = _cache_key(
            model=None,
            sample_id=sample_id,
            instruction=instruction,
            source_identity=str(source_ref),
            source_path=None,
            output_path=output_path,
        )
        if cache_key in cached:
            score_row = cached[cache_key]
        else:
            source_row = image_loader.get_row(dataset_id, split, row_index)
            if (
                row_index != metadata.get("row_index")
                or source_row.get("task") != task
                or source_row.get("image_path") != metadata.get("image_path")
                or source_row.get("instruction") != instruction
            ):
                raise ValueError(f"Dataset/annotation mismatch for {sample_id}")
            scores = _score(
                scorer,
                image_loader.as_image(source_row[image_field]),
                output_path,
                instruction,
            )
            output_stat = output_path.stat()
            score_row = {
                "sample_id": sample_id,
                "record_position": index,
                "dataset_row_index": metadata.get("row_index"),
                "task": task,
                "prompt_mode": prompt_mode,
                "instruction": instruction,
                "output_path": str(output_path),
                "output_size_bytes": output_stat.st_size,
                "output_mtime_ns": output_stat.st_mtime_ns,
                "cache_key": cache_key,
                **scores,
            }
        rows.append(score_row)
        _write_jsonl(metric_path, rows)
    summary = _save_results(rows, output_dir, "editscore")
    return rows, summary


def evaluate_replicate_compbench_outputs(
    replicate_root: Path,
    *,
    scorer: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score successful per-model Replicate CompBench outputs with EditScore."""
    root = replicate_root.expanduser().resolve()
    saved_outputs = collect_replicate_saved_outputs(root)
    if scorer is None:
        scorer = create_editscore_scorer(
            num_pass=int(os.environ.get("EDITSCORE_NUM_PASS", "1")),
            score_range=int(os.environ.get("EDITSCORE_SCORE_RANGE", "25")),
        )
    metric_path = root / "metrics" / "editscore.jsonl"
    cached = _read_cached_rows(metric_path)
    rows: list[dict[str, Any]] = []
    for saved in tqdm(saved_outputs, desc="Replicate EditScore", unit="sample"):
        model = saved["model"]
        sample_id = saved["sample_id"]
        instruction = saved["instruction"]
        source_path: Path = saved["source_path"]
        output_path: Path = saved["output_path"]
        cache_key = _cache_key(
            model=model,
            sample_id=sample_id,
            instruction=instruction,
            source_identity=str(source_path),
            source_path=source_path,
            output_path=output_path,
        )
        if cache_key in cached:
            score_row = cached[cache_key]
        else:
            scores = _score(
                scorer,
                _load_rgb_image(source_path),
                output_path,
                instruction,
            )
            output_stat = output_path.stat()
            source_stat = source_path.stat()
            score_row = {
                "model": model,
                "sample_id": sample_id,
                "task": saved["task"],
                "instruction": instruction,
                "source_path": str(source_path),
                "source_size_bytes": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "output_path": str(output_path),
                "output_size_bytes": output_stat.st_size,
                "output_mtime_ns": output_stat.st_mtime_ns,
                "cache_key": cache_key,
                **scores,
            }
        rows.append(score_row)
        _write_jsonl(metric_path, rows)
    summary = _save_results(rows, root, "editscore")
    return rows, summary
