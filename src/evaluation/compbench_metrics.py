from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .common import HFImageLoader, read_jsonl

PromptMode = Literal["text", "mixed"]
METRIC_NAMES = ("LC-T", "LC-I", "PSNR", "SSIM", "LPIPS")
IMAGE_SIZE = (512, 512)


def _safe_sample_id(sample_id: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"


def collect_saved_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    prompt_mode: PromptMode,
    task_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Join saved outputs to annotation-list positions using stable sample IDs.

    Inference manifests may store annotation_index as the original dataset row,
    which can be sparse and is not necessarily an index into ``records``.
    """
    results_path = output_dir / "results" / "results.jsonl"
    images_dir = output_dir / "results" / "images"
    positions_by_sample_id: dict[str, int] = {}
    for position, record in enumerate(records):
        metadata = record.get("metadata") or {}
        if task_filter is not None and metadata.get("task") not in task_filter:
            continue
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Annotation at list position {position} has no sample_id")
        if sample_id in positions_by_sample_id:
            raise ValueError(f"Duplicate annotation sample_id: {sample_id!r}")
        positions_by_sample_id[sample_id] = position

    positions_by_slug: dict[str, int] = {}
    for sample_id, position in positions_by_sample_id.items():
        slug = _safe_sample_id(sample_id)
        if slug in positions_by_slug:
            raise ValueError(
                f"Annotation sample IDs collide after filename normalization: "
                f"{sample_id!r} and "
                f"{records[positions_by_slug[slug]].get('sample_id')!r}"
            )
        positions_by_slug[slug] = position
    outputs: dict[int, dict[str, Any]] = {}
    if results_path.is_file():
        for result in read_jsonl(results_path):
            if result.get("status") != "completed":
                continue
            if result.get("prompt_mode") != prompt_mode:
                raise ValueError(
                    f"Expected {prompt_mode!r} results in {results_path}, got "
                    f"{result.get('prompt_mode')!r} for {result.get('sample_id')!r}"
                )
            sample_id = result.get("sample_id")
            if (
                not isinstance(sample_id, str)
                or sample_id not in positions_by_sample_id
            ):
                if task_filter is not None:
                    continue
                raise ValueError(
                    f"Result sample_id is not in the selected annotations: "
                    f"{sample_id!r}"
                )
            position = positions_by_sample_id[sample_id]
            output_path = result.get("output_path")
            if not isinstance(output_path, str) or not output_path:
                raise ValueError(
                    f"Completed result has no output_path in {results_path}: "
                    f"{result.get('sample_id')!r}"
                )
            path = Path(output_path).expanduser()
            if path.is_file():
                outputs[position] = {
                    "record_position": position,
                    "sample_id": sample_id,
                    "output_path": path.resolve(),
                }

    for path in sorted(images_dir.glob("*_output.png")):
        prefix, separator, _ = path.name.partition("_")
        if not separator or not prefix.isdigit():
            continue
        filename_body = path.name[len(prefix) + 1 :]
        suffix = "_output.png"
        if not filename_body.endswith(suffix):
            continue
        slug = filename_body[: -len(suffix)]
        position = positions_by_slug.get(slug)
        if position is None:
            continue
        outputs.setdefault(
            position,
            {
                "record_position": position,
                "sample_id": records[position].get("sample_id"),
                "output_path": path.resolve(),
            },
        )
    if not outputs:
        if task_filter is not None:
            raise FileNotFoundError(
                f"No saved output images for tasks {sorted(task_filter)} found "
                f"in {images_dir}"
            )
        raise FileNotFoundError(f"No saved output images found in {images_dir}")
    return [outputs[position] for position in sorted(outputs)]


def _captions(value: Any) -> list[str]:
    if isinstance(value, str):
        captions = [caption.strip() for caption in value.split("|")]
    elif isinstance(value, list):
        captions = [caption.strip() for caption in value if isinstance(caption, str)]
    else:
        captions = []
    return [caption for caption in captions if caption]


def _load_mask(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("L")
    if isinstance(value, dict):
        return HFImageLoader.as_image(value).convert("L")
    raise TypeError(f"Unsupported CompBench mask value: {type(value).__name__}")


def _resize_image(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BICUBIC)


def _resize_mask(mask: Image.Image) -> Image.Image:
    return (
        mask.convert("L")
        .resize(IMAGE_SIZE, Image.Resampling.NEAREST)
        .point(lambda value: 255 if value > 0 else 0)
    )


def _load_metric_models(device: torch.device, clip_model_name: str) -> dict[str, Any]:
    try:
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError(
            "CompBench metrics require torchmetrics and transformers. "
            "Install them in the notebook's selected environment."
        ) from exc

    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }


def _clip_features(output: Any) -> torch.Tensor:
    """Get projected CLIP embeddings across Transformers return conventions."""
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    raise TypeError(
        "CLIP feature method must return a Tensor or an object with tensor "
        f"pooler_output, got {type(output).__name__}"
    )


def _score_pair(
    source: Image.Image,
    generated: Image.Image,
    target: Image.Image,
    mask: Image.Image,
    captions: list[str],
    *,
    models: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], tuple[int, int, int, int]]:
    source = _resize_image(source)
    generated = _resize_image(generated)
    target = _resize_image(target)
    mask = _resize_mask(mask)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("CompBench edit mask is empty")
    if not captions:
        raise ValueError("CompBench metadata.caption is empty")

    def to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    source_tensor = to_tensor(source)
    generated_tensor = to_tensor(generated)
    background = torch.from_numpy(
        1.0 - np.asarray(mask, dtype=np.float32)[None, None] / 255.0
    ).to(device)
    source_background = source_tensor * background
    generated_background = generated_tensor * background

    with torch.inference_mode():
        psnr = models["psnr"](generated_background, source_background).item()
        ssim = models["ssim"](generated_background, source_background).item()
        lpips = models["lpips"](
            generated_background * 2.0 - 1.0,
            source_background * 2.0 - 1.0,
        ).item()

        generated_crop = generated.crop(bbox)
        text_scores = []
        for caption in captions:
            text_inputs = models["clip_processor"](
                images=[generated_crop],
                text=[caption],
                return_tensors="pt",
                padding=True,
            ).to(device)
            image_features = _clip_features(
                models["clip_model"].get_image_features(
                    pixel_values=text_inputs["pixel_values"]
                )
            )
            text_features = _clip_features(
                models["clip_model"].get_text_features(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs.get("attention_mask"),
                )
            )
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            text_scores.append(
                (100.0 * (image_features * text_features).sum(dim=-1)).mean().item()
            )

        target_crop = target.crop(bbox)
        clip_inputs = models["clip_processor"](
            images=[generated_crop, target_crop],
            return_tensors="pt",
            padding=True,
        ).to(device)
        features = _clip_features(
            models["clip_model"].get_image_features(**clip_inputs)
        )
        image_similarity = F.cosine_similarity(
            features[0].unsqueeze(0), features[1].unsqueeze(0)
        ).item()

    scores = {
        "LC-T": float(fmean(text_scores)),
        "LC-I": float(image_similarity),
        "PSNR": float(psnr),
        "SSIM": float(ssim),
        "LPIPS": float(lpips),
    }
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError(f"Non-finite CompBench metric: {scores}")
    return scores, bbox


def _summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    grouped["overall"] = rows
    summary: dict[str, dict[str, float | int]] = {}
    for task, task_rows in sorted(grouped.items()):
        if not task_rows:
            continue
        summary[task] = {
            metric: fmean(float(row[metric]) for row in task_rows)
            for metric in METRIC_NAMES
        }
        summary[task]["count"] = len(task_rows)
    return summary


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


def evaluate_saved_compbench_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    image_loader: HFImageLoader,
    prompt_mode: PromptMode,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    device_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    """Apply the released CompBench metrics to one saved inference run."""
    if prompt_mode not in ("text", "mixed"):
        raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")
    output_dir = output_dir.expanduser().resolve()
    saved_outputs = collect_saved_outputs(records, output_dir, prompt_mode=prompt_mode)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = _load_metric_models(device, clip_model_name)
    metric_path = output_dir / "metrics" / "compbench_metrics.jsonl"

    cached: dict[tuple[Any, ...], dict[str, Any]] = {}
    if metric_path.is_file():
        for row in read_jsonl(metric_path):
            key = (
                row.get("sample_id"),
                row.get("output_path"),
                row.get("output_size_bytes"),
                row.get("output_mtime_ns"),
                row.get("task"),
                tuple(row.get("captions", [])),
                row.get("prompt_mode"),
            )
            cached[key] = row

    rows: list[dict[str, Any]] = []
    for saved in tqdm(saved_outputs, desc="CompBench metrics", unit="sample"):
        index = saved["record_position"]
        record = records[index]
        metadata = record.get("metadata") or {}
        task = metadata.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"Missing metadata.task for {record.get('sample_id')}")
        output_path: Path = saved["output_path"]
        stat = output_path.stat()
        cache_key = (
            record.get("sample_id"),
            str(output_path),
            stat.st_size,
            stat.st_mtime_ns,
            task,
            tuple(_captions(metadata.get("caption"))),
            prompt_mode,
        )
        if cache_key in cached:
            row = cached[cache_key]
        else:
            source_reference = image_loader.parse_reference(record.get("source_ref"))
            target_reference = image_loader.parse_reference(record.get("target_ref"))
            if source_reference is None or target_reference is None:
                raise ValueError(
                    f"Missing CompBench source_ref or target_ref for "
                    f"{record.get('sample_id')}"
                )
            dataset_id, split, source_index, source_field = source_reference
            source_row = image_loader.get_row(dataset_id, split, source_index)
            target_dataset, target_split, target_index, target_field = target_reference
            if (target_dataset, target_split, target_index) == (
                dataset_id,
                split,
                source_index,
            ):
                target_row = source_row
            else:
                target_row = image_loader.get_row(
                    target_dataset, target_split, target_index
                )
            if target_dataset != dataset_id or target_split != split:
                raise ValueError(
                    f"Dataset mismatch for CompBench sample {record.get('sample_id')}"
                )
            captions = _captions(metadata.get("caption"))
            if (
                source_index != metadata.get("row_index")
                or source_row.get("task") != task
                or source_row.get("image_path") != metadata.get("image_path")
                or _captions(source_row.get("caption")) != captions
            ):
                raise ValueError(
                    f"Dataset/annotation mismatch for CompBench sample "
                    f"{record.get('sample_id')}"
                )
            with Image.open(output_path) as generated_file:
                scores, bbox = _score_pair(
                    image_loader.as_image(source_row[source_field]),
                    generated_file.copy(),
                    image_loader.as_image(target_row[target_field]),
                    _load_mask(source_row["mask"]),
                    captions,
                    models=models,
                    device=device,
                )
            row = {
                "sample_id": record.get("sample_id"),
                "record_position": index,
                "dataset_row_index": metadata.get("row_index"),
                "task": task,
                "prompt_mode": prompt_mode,
                "captions": captions,
                "output_path": str(output_path),
                "output_size_bytes": stat.st_size,
                "output_mtime_ns": stat.st_mtime_ns,
                "edit_bbox_xyxy": list(bbox),
                **scores,
            }
        rows.append(row)
        if len(rows) % 25 == 0:
            _write_jsonl(metric_path, rows)

    _write_jsonl(metric_path, rows)
    summary = _summary(rows)
    summary_path = output_dir / "metrics" / "compbench_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Scored {len(rows)} saved CompBench outputs.")
    print(f"Per-sample metrics: {metric_path}")
    print(f"Task and overall summary: {summary_path}")
    return rows, summary
