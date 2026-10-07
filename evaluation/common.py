from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import time
import traceback
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

AGENT_EXECUTION_TIMEOUT_SECONDS = 120
DEFAULT_PROMPT_SELECTION_SEED = 42


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                yield value


class HFImageLoader:
    """Resolve hf://dataset/split/original_row_index/field references."""

    def __init__(self) -> None:
        self.streams: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def parse_reference(
        reference: str | None,
    ) -> tuple[str, str, int, str] | None:
        if reference is None:
            return None
        if not isinstance(reference, str) or not reference.startswith("hf://"):
            raise ValueError(f"Unsupported image reference: {reference!r}")
        parts = reference.removeprefix("hf://").split("/")
        if len(parts) < 4:
            raise ValueError(f"Invalid Hugging Face image reference: {reference}")
        try:
            row_index = int(parts[-2])
        except ValueError as exc:
            raise ValueError(
                f"Invalid row index in image reference: {reference}"
            ) from exc
        if row_index < 0:
            raise ValueError(f"Row index must be non-negative: {reference}")
        return "/".join(parts[:-3]), parts[-3], row_index, parts[-1]

    def get_row(self, dataset_id: str, split: str, row_index: int) -> dict[str, Any]:
        from datasets import load_dataset

        key = (dataset_id, split)
        state = self.streams.get(key)
        if state is not None and row_index == state["last_index"]:
            return state["last_row"]
        if state is None or row_index < state["next_index"]:
            dataset = load_dataset(dataset_id, split=split, streaming=True)
            state = {
                "iterator": iter(dataset),
                "next_index": 0,
                "last_index": None,
                "last_row": None,
            }
            self.streams[key] = state
        while state["next_index"] <= row_index:
            try:
                row = next(state["iterator"])
            except StopIteration as exc:
                raise IndexError(
                    f"Row {row_index} is unavailable in {dataset_id}/{split}"
                ) from exc
            state["last_index"] = state["next_index"]
            state["last_row"] = row
            state["next_index"] += 1
        return state["last_row"]

    @staticmethod
    def as_image(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            image_path = value.get("path")
            if image_bytes:
                with Image.open(BytesIO(image_bytes)) as image:
                    return image.convert("RGB")
            if image_path and Path(image_path).is_file():
                with Image.open(image_path) as image:
                    return image.convert("RGB")
        raise TypeError(f"Cannot decode dataset image value: {type(value).__name__}")

    def load_pair(
        self, record: dict[str, Any]
    ) -> tuple[Image.Image, Image.Image | None]:
        source_ref = self.parse_reference(record.get("source_ref"))
        target_ref = self.parse_reference(record.get("target_ref"))
        if source_ref is None:
            raise ValueError(f"No source_ref for sample {record.get('sample_id')}")
        dataset_id, split, row_index, source_field = source_ref
        source_row = self.get_row(dataset_id, split, row_index)
        source_image = self.as_image(source_row[source_field])

        target_image = None
        if target_ref is not None:
            target_dataset, target_split, target_index, target_field = target_ref
            if (target_dataset, target_split, target_index) == (
                dataset_id,
                split,
                row_index,
            ):
                target_row = source_row
            else:
                target_row = self.get_row(target_dataset, target_split, target_index)
            target_image = self.as_image(target_row[target_field])
        return source_image, target_image


def _get_instruction(record: dict[str, Any], benchmark: str) -> str | None:
    fields = (
        ("mixed_instruction", "instruction", "prompt", "task")
        if benchmark == "compbench"
        else (
            "original_instruction",
            "instruction",
            "prompt",
            "edit_instruction",
            "mixed_instruction",
            "task",
        )
    )
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_sample(
    record: dict[str, Any],
    *,
    benchmark: str,
    image_loader: HFImageLoader,
) -> dict[str, Any]:
    source_image, target_image = image_loader.load_pair(record)
    references = []
    for ref in record.get("references", []):
        prompts = ref.get("visual_prompts") or {}
        scribble_info = prompts.get("scribble") or {}
        strokes = scribble_info.get("strokes") or []
        references.append(
            {
                "token": ref.get("token"),
                "role": ref.get("role"),
                "span": ref.get("span"),
                "bbox_xyxy": (prompts.get("box") or {}).get("xyxy"),
                "point_xy": (prompts.get("point") or {}).get("xy"),
                "scribble_strokes": strokes,
            }
        )
    return {
        "sample_id": record.get("sample_id"),
        "instruction": _get_instruction(record, benchmark),
        "source_image": source_image,
        "target_image": target_image,
        "references": references,
        "record": record,
    }


def build_agent_task(sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return only the benchmark's user instruction and bound reference objects."""
    objects: dict[str, Any] = {"image_0": sample["source_image"]}
    for ref in sample["references"]:
        selected = ref.get("selected_prompt")
        if selected is not None:
            objects[selected["variable"]] = selected["value"]
    return sample["instruction"], objects


def build_agent_context(sample: dict[str, Any]) -> str:
    """Build agent guidance separately from the user instruction/observation."""
    lines = [
        "Use image_0 as the source image. Follow the user instruction and return the "
        "final edited image.",
        "For each referenced object, call segment_object with its single supplied "
        "box, point, or scribble selector to create a mask before editing. Never use "
        "annotation masks or crops. Do not combine selector types for one reference. "
        "Apply edits to each required object in sequence and preserve unrelated "
        "image content.",
        "Choose edit_object mode intentionally: local for appearance-only changes, "
        "regional for identity/shape/accessory changes near the object, global only "
        "when surrounding scene changes are requested. Its prompt must describe the "
        "requested transformation, not just the desired category: for example, for "
        "'Let the man with longer curly hair, wearing a loose tie be a woman,' say "
        "'Transform the selected man into a woman, keeping his longer curly hair, "
        "loose tie, and pose.' Include all requested attributes and preserve "
        "unmentioned details. Use other registered tools when they better match the "
        "requested operation.",
    ]
    for index, ref in enumerate(sample["references"]):
        token = ref["token"] or f"reference_{index}"
        selected = ref.get("selected_prompt")
        description = ref.get("span")
        description_line = (
            f" refers to {description!r};" if description else ""
        )
        selector_line = ""
        if selected is not None:
            selector_line = (
                f"{selected['type']} selector variable {selected['variable']}"
            )
        if selector_line:
            lines.append(
                f"Reference {token}{description_line} use {selector_line}."
            )
    lines.append("Do not invent selector coordinates.")
    return "\n".join(lines)


def select_reference_prompts(
    sample: dict[str, Any],
    *,
    seed: int = DEFAULT_PROMPT_SELECTION_SEED,
) -> None:
    """Select one available visual selector per reference reproducibly."""
    rng = random.Random(f"{seed}:{sample.get('sample_id')}")
    for index, ref in enumerate(sample["references"]):
        prompt_options = []
        if ref["bbox_xyxy"] is not None:
            prompt_options.append(("box", ref["bbox_xyxy"]))
        if ref["point_xy"] is not None:
            prompt_options.append(("point", ref["point_xy"]))
        strokes = ref["scribble_strokes"]
        if strokes and strokes[0]:
            prompt_options.append(("scribble", strokes[0]))

        if prompt_options:
            prompt_type, value = rng.choice(prompt_options)
            ref["selected_prompt"] = {
                "type": prompt_type,
                "value": value,
                "variable": f"{prompt_type}_{index}",
            }
        else:
            ref["selected_prompt"] = None


def annotate_sample(sample: dict[str, Any]) -> Image.Image:
    """Draw each reference's selected prompt over a copy of the source image."""
    annotated = sample["source_image"].convert("RGB").copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    width, height = annotated.size
    line_width = max(2, round(min(width, height) / 256))
    radius = max(4, line_width * 2)

    for index, ref in enumerate(sample["references"]):
        selected = ref.get("selected_prompt")
        if not selected:
            continue
        prompt_type = selected["type"]
        value = selected["value"]
        color = (255, 64, 64, 255)
        label_origin = (0, 0)
        if prompt_type == "box":
            x0, y0, x1, y1 = (round(float(point)) for point in value)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
            label_origin = (x0, y0)
        elif prompt_type == "point":
            x, y = (round(float(point)) for point in value)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 220, 0, 220),
                outline=(0, 0, 0, 255),
                width=max(1, line_width // 2),
            )
            label_origin = (x, y)
        elif prompt_type == "scribble":
            points = [(round(float(x)), round(float(y))) for x, y in value]
            if len(points) == 1:
                x, y = points[0]
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            elif points:
                draw.line(points, fill=(0, 220, 255, 255), width=line_width * 2)
                label_origin = points[0]
        label = ref.get("token") or f"reference_{index}"
        x, y = label_origin
        draw.text(
            (x, max(0, y - line_width * 8)),
            f"{label}: {prompt_type}",
            fill=color,
        )
    return annotated


def json_safe(
    value: Any, asset_dir: Path, prefix: str = "raw", seen: set[int] | None = None
) -> Any:
    """JSON encode agent traces, saving binary/image/tensor payloads as assets."""
    if seen is None:
        seen = set()
    filename_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("._")
    filename_prefix = filename_prefix[:100] or "asset"
    filename_prefix += "_" + hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:10]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    value_id = id(value)
    if value_id in seen:
        return {"__type__": type(value).__name__, "__repr__": "<recursive reference>"}
    if isinstance(value, Image.Image):
        asset_dir.mkdir(parents=True, exist_ok=True)
        asset_path = asset_dir / f"{filename_prefix}.png"
        value.save(asset_path)
        return {
            "__type__": "PIL.Image",
            "path": str(asset_path),
            "mode": value.mode,
            "size": list(value.size),
        }
    if isinstance(value, bytes):
        asset_dir.mkdir(parents=True, exist_ok=True)
        asset_path = asset_dir / f"{filename_prefix}.bin"
        asset_path.write_bytes(value)
        return {"__type__": "bytes", "path": str(asset_path), "length": len(value)}

    seen.add(value_id)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "__type__": type(value).__name__,
                **{
                    field.name: json_safe(
                        getattr(value, field.name),
                        asset_dir,
                        f"{prefix}_{field.name}",
                        seen,
                    )
                    for field in dataclasses.fields(value)
                },
            }
        if isinstance(value, dict):
            return {
                str(key): json_safe(item, asset_dir, f"{prefix}_{key}", seen)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [
                json_safe(item, asset_dir, f"{prefix}_{index}", seen)
                for index, item in enumerate(value)
            ]
        if isinstance(value, np.ndarray):
            asset_dir.mkdir(parents=True, exist_ok=True)
            asset_path = asset_dir / f"{filename_prefix}.npy"
            np.save(asset_path, value, allow_pickle=False)
            return {
                "__type__": "numpy.ndarray",
                "path": str(asset_path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            tensor = value.detach().cpu()
            asset_dir.mkdir(parents=True, exist_ok=True)
            asset_path = asset_dir / f"{filename_prefix}.npy"
            np.save(asset_path, tensor.numpy(), allow_pickle=False)
            return {
                "__type__": type(value).__name__,
                "path": str(asset_path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        if hasattr(value, "model_dump"):
            return {
                "__type__": type(value).__name__,
                "data": json_safe(
                    value.model_dump(), asset_dir, prefix, seen
                ),
            }
        if hasattr(value, "to_dict"):
            return {
                "__type__": type(value).__name__,
                "data": json_safe(value.to_dict(), asset_dir, prefix, seen),
            }
        if hasattr(value, "__dict__"):
            return {
                "__type__": type(value).__name__,
                "data": json_safe(vars(value), asset_dir, prefix, seen),
            }
        return {"__type__": type(value).__name__, "__repr__": repr(value)}
    finally:
        seen.discard(value_id)


def run_one(
    record: dict[str, Any],
    annotation_index: int,
    *,
    benchmark: str,
    output_dir: Path,
    image_loader: HFImageLoader,
    model: Any,
    tools: list[Any],
    seed: int = DEFAULT_PROMPT_SELECTION_SEED,
    max_steps: int = 16,
) -> dict[str, Any]:
    from core import Agent, local_executor

    start = time.perf_counter()
    sample_id = record.get("sample_id") or f"annotation_{annotation_index}"
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
    file_stem = f"{annotation_index:06d}_{safe_id}"
    log_dir = output_dir / "logs" / file_stem
    images_dir = output_dir / "results" / "images"
    log_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "benchmark": benchmark,
        "sample_id": sample_id,
        "annotation_index": annotation_index,
        "instruction": _get_instruction(record, benchmark),
        "target_available": record.get("target_ref") is not None,
        "random_seed": seed,
        "status": "running",
        "log_dir": str(log_dir),
    }
    agent = None
    result = None
    sample = None
    try:
        sample = load_sample(
            record,
            benchmark=benchmark,
            image_loader=image_loader,
        )
        if not sample["instruction"]:
            raise ValueError(
                f"Record has no supported instruction field for {benchmark}."
            )
        sample["sample_id"] = sample.get("sample_id") or sample_id
        entry["target_available"] = sample["target_image"] is not None
        select_reference_prompts(sample, seed=seed)
        task, objects = build_agent_task(sample)
        agent_context = build_agent_context(sample)
        entry["reference_prompts"] = [
            {
                "token": ref["token"],
                "type": ref["selected_prompt"]["type"],
                "value": ref["selected_prompt"]["value"],
            }
            for ref in sample["references"]
            if ref.get("selected_prompt") is not None
        ]
        source_path = images_dir / f"{file_stem}_source.png"
        annotated_path = images_dir / f"{file_stem}_annotated.png"
        sample["source_image"].save(source_path)
        annotate_sample(sample).save(annotated_path)
        entry.update(
            {
                "source_path": str(source_path),
                "annotated_path": str(annotated_path),
            }
        )
        agent = Agent(
            model=model,
            tools=tools,
            max_steps=max_steps,
            instructions=agent_context,
            executor=local_executor(timeout_seconds=AGENT_EXECUTION_TIMEOUT_SECONDS),
            verbosity_level=1,
        )
        result = agent.send(task, objects=objects)
        images = [
            obj for obj in result.message.objects if isinstance(obj, Image.Image)
        ]
        if not images:
            raise RuntimeError(
                "Agent completed without returning a Pillow image in "
                "send_message.objects."
            )

        output_path = images_dir / f"{file_stem}_output.png"
        images[0].convert("RGB").save(output_path)
        target_path = None
        if sample["target_image"] is not None:
            target_path = images_dir / f"{file_stem}_target.png"
            sample["target_image"].save(target_path)
        entry.update(
            {
                "status": "completed",
                "message": result.message.text,
                "output_path": str(output_path),
                "target_path": str(target_path) if target_path else None,
            }
        )
    except Exception as exc:
        entry.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(entry["traceback"])
    finally:
        entry["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        trajectory_text_path = log_dir / "trajectory.txt"
        trajectory_json_path = log_dir / "trajectory.json"
        variables_json_path = log_dir / "variables.json"
        raw_json_path = log_dir / "raw.json"
        entry.update(
            {
                "trajectory_text_path": str(trajectory_text_path),
                "trajectory_json_path": str(trajectory_json_path),
                "variables_path": str(variables_json_path),
                "raw_path": str(raw_json_path),
            }
        )
        try:
            trajectory = agent.trajectory if agent is not None else []
            trajectory_text = agent.trajectory_text if agent is not None else ""
            raw_steps = agent.raw_steps if agent is not None else []
            variables = agent.variables if agent is not None else {}
            trajectory_text_path.write_text(trajectory_text, encoding="utf-8")
            assets_dir = log_dir / "debug_assets"
            serialized_variables = json_safe(
                variables, assets_dir, "variables"
            )
            variables_json_path.write_text(
                json.dumps(
                    serialized_variables,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            trajectory_json_path.write_text(
                json.dumps(
                    json_safe(trajectory, assets_dir, "trajectory"),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raw_payload = {
                "benchmark": benchmark,
                "sample_id": sample_id,
                "annotation_index": annotation_index,
                "model_id": getattr(model, "model_id", None),
                "instruction": entry["instruction"],
                "reference_prompts": entry.get("reference_prompts", []),
                "random_seed": seed,
                "status": entry["status"],
                "error": entry.get("error"),
                "traceback": entry.get("traceback"),
                "agent_trajectory": trajectory,
                "agent_raw_steps": raw_steps,
                "agent_variables": serialized_variables,
                "turn_result_raw": result.raw if result is not None else None,
                "message": result.message.text if result is not None else None,
                "reference_record": record,
            }
            raw_json_path.write_text(
                json.dumps(
                    json_safe(raw_payload, assets_dir, "raw"),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as save_error:
            entry["debug_save_error"] = repr(save_error)
            print(
                f"Failed to save trajectory/raw debug files for "
                f"{sample_id}: {save_error!r}"
            )
    return entry


def run_benchmark(
    records: list[dict[str, Any]],
    sample_indices: Iterable[int],
    *,
    benchmark: str,
    output_dir: Path,
    image_loader: HFImageLoader,
    model: Any,
    tools: list[Any],
    seed: int = DEFAULT_PROMPT_SELECTION_SEED,
    display_fn: Any = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    results_path = output_dir / "results" / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as stream:
        for annotation_index in sample_indices:
            if not 0 <= annotation_index < len(records):
                raise IndexError(f"Annotation index out of range: {annotation_index}")
            result = run_one(
                records[annotation_index],
                annotation_index,
                benchmark=benchmark,
                output_dir=output_dir,
                image_loader=image_loader,
                model=model,
                tools=tools,
                seed=seed,
            )
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                result["sample_id"],
                result["status"],
                result["elapsed_seconds"],
                "seconds",
            )
            if result["status"] == "completed" and display_fn is not None:
                display_fn(Image.open(result["output_path"]))
    print("Saved run summary:", results_path)
    return results
