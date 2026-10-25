from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import shutil
import time
import traceback
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from framework.utils.randomness import seed_random_generators

from .first_turn import (
    FirstTurnConfig,
    FirstTurnEntry,
    prompt_signature,
    task_fingerprint,
)
from .spatial import needs_spatial_anchor_box

if TYPE_CHECKING:
    from framework.core.agent import FirstTurnCommand

AGENT_EXECUTION_TIMEOUT_SECONDS = 300
PromptMode = Literal["text", "mixed"]


def _validate_prompt_mode(prompt_mode: str) -> None:
    if prompt_mode not in ("text", "mixed"):
        raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")


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


class MiraBenchImageLoader:
    """Load file-based MIRA-Bench images from a local dataset snapshot."""

    _REFERENCE_PREFIXES = (
        "mira_bench://benchmark/",
        "hf://ziqiangoodgood/MIRAGE/benchmark/",
    )

    def __init__(self, image_root: str | Path) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"MIRA-Bench image directory not found: {self.image_root}"
            )

    def load_pair(
        self, record: dict[str, Any]
    ) -> tuple[Image.Image, Image.Image | None]:
        if record.get("target_ref") is not None:
            raise ValueError("MIRA-Bench annotations must not include target_ref")
        reference = record.get("source_ref")
        if not isinstance(reference, str):
            raise ValueError(f"No source_ref for sample {record.get('sample_id')}")
        relative_path = next(
            (
                reference.removeprefix(prefix)
                for prefix in self._REFERENCE_PREFIXES
                if reference.startswith(prefix)
            ),
            None,
        )
        if not relative_path:
            raise ValueError(f"Unsupported MIRA-Bench image reference: {reference!r}")
        image_path = Path(relative_path)
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError(
                f"MIRA-Bench image reference escapes dataset directory: {reference!r}"
            )
        # Don't resolve symlinks: HF snapshot files link into a separate blobs dir.
        # Containment is already guaranteed by rejecting absolute and ".." paths.
        resolved_path = self.image_root / image_path
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"MIRA-Bench image not found for sample "
                f"{record.get('sample_id')}: {resolved_path}"
            )
        with Image.open(resolved_path) as image:
            source_image = image.convert("RGB")
        return source_image, None


def _get_instruction(record: dict[str, Any], prompt_mode: PromptMode) -> str | None:
    _validate_prompt_mode(prompt_mode)
    primary_field = (
        "original_instruction" if prompt_mode == "text" else "mixed_instruction"
    )
    value = record.get(primary_field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_sample(
    record: dict[str, Any],
    *,
    benchmark: str,
    image_loader: HFImageLoader | MiraBenchImageLoader,
    prompt_mode: PromptMode = "mixed",
) -> dict[str, Any]:
    _validate_prompt_mode(prompt_mode)
    source_image, target_image = image_loader.load_pair(record)
    metadata = record.get("metadata") or {}
    spatial_destination = benchmark == "compbench" and metadata.get("task") in {
        "add",
        "multi_object_add",
    }
    references = []
    for ref in record.get("references", []):
        region = ref.get("region") or {}
        prompts = ref.get("visual_prompts") or region.get("visual_prompts") or {}
        scribble_info = prompts.get("scribble") or {}
        strokes = scribble_info.get("strokes") or []
        references.append(
            {
                "token": ref.get("token"),
                "role": ref.get("role"),
                "span": ref.get("span"),
                "bbox_xyxy": (prompts.get("box") or {}).get("xyxy")
                or region.get("bbox_xyxy"),
                "point_xy": (prompts.get("point") or {}).get("xy"),
                "scribble_strokes": strokes,
                "spatial_destination": spatial_destination,
            }
        )
    return {
        "sample_id": record.get("sample_id"),
        "instruction": _get_instruction(record, prompt_mode),
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


def build_agent_context(
    sample: dict[str, Any], prompt_mode: PromptMode = "mixed"
) -> str:
    """Build agent guidance separately from the user instruction/observation."""
    _validate_prompt_mode(prompt_mode)
    lines = [
        "Use image_0 as the source image. Follow the user instruction and return the "
        "final edited image.",
        (
            "For each referenced existing entity, use segment_object with its "
            "natural-language description. No visual selector is supplied for "
            "these references; do not assume annotation boxes, points, scribbles, "
            "masks, or crops are available. For a spatial destination without a "
            "supplied box, infer placement from the image and instruction."
            if prompt_mode == "text"
            else (
                "When the instruction contains a visual placeholder such as "
                "<point_0>, <box_1>, or <scribble_2>, use the exactly matching "
                "variable bound in the observation; do not infer a variable from "
                "placeholder order. For each reference to an existing entity, "
                "call segment_object with its single supplied selector to create "
                "a mask before editing. A spatial destination is not an existing "
                "entity: use its supplied box directly with add_object, without "
                "calling segment_object. Never use annotation masks or crops. Do "
                "not combine selector types for one reference."
            )
        )
        + " Apply edits to each required object in sequence and preserve unrelated "
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
    spatial_anchor_needed = needs_spatial_anchor_box(sample.get("instruction") or "")
    if spatial_anchor_needed:
        lines.append(
            "For relative placement without a supplied destination box, use the "
            "anchor's full bounding box and image dimensions to calculate an "
            "approximate destination [x1, y1, x2, y2] in pixels; clamp it to the "
            "image and pass it as add_object(box=...). When no selector is "
            "available for the anchor, segment it and use its mask bounds."
        )
    if prompt_mode == "mixed" and spatial_anchor_needed:
        lines.append(
            "This request places a new object relative to an existing reference. "
            "Use the reference's supplied box as the full-object anchor when "
            "calculating placement; do not infer the object's full extent from a "
            "point-selected or segmented mask, which may cover only a visible part. "
            "Use segment_object with the supplied box if a mask is needed."
        )
    for index, ref in enumerate(sample["references"]):
        token = ref.get("display_token") or ref["token"] or f"reference_{index}"
        selected = ref.get("selected_prompt")
        description = ref.get("span")
        description_line = (
            f" refers to {description!r};"
            if description and (prompt_mode == "text" or ref.get("spatial_destination"))
            else ""
        )
        if prompt_mode == "text":
            if description:
                lines.append(
                    f"Reference {token}{description_line} use that natural-language "
                    "description with segment_object(text=...)."
                )
            continue
        selector_line = ""
        if selected is not None:
            selector_line = (
                f"{selected['type']} selector variable {selected['variable']}"
            )
        if selector_line:
            if ref.get("spatial_destination"):
                lines.append(
                    f"Reference {token}{description_line} denotes a spatial "
                    f"destination, not an entity to select; use {selector_line} "
                    "directly as the box for add_object."
                )
            else:
                lines.append(
                    f"Reference {token}{description_line} use {selector_line}."
                )
    lines.append("Do not invent selector coordinates.")
    return "\n".join(lines)


def select_reference_prompts(
    sample: dict[str, Any],
    prompt_mode: PromptMode = "mixed",
) -> None:
    """Select a visual selector per reference, unless running text-only mode."""
    _validate_prompt_mode(prompt_mode)
    if prompt_mode == "text":
        sample.pop("annotation_instruction", None)
        for ref in sample["references"]:
            ref["selected_prompt"] = None
            ref["display_token"] = ref.get("token")
        return

    sample.setdefault("annotation_instruction", sample.get("instruction") or "")
    instruction = sample["annotation_instruction"]
    display_tokens: dict[str, str] = {}
    seed = seed_random_generators()
    rng = random.Random(f"{seed}:{sample.get('sample_id')}")
    use_anchor_boxes = needs_spatial_anchor_box(sample.get("instruction") or "")
    for index, ref in enumerate(sample["references"]):
        prompt_options = []
        if ref["bbox_xyxy"] is not None:
            prompt_options.append(("box", ref["bbox_xyxy"]))
        if not ref.get("spatial_destination"):
            if ref["point_xy"] is not None:
                prompt_options.append(("point", ref["point_xy"]))
            strokes = ref["scribble_strokes"]
            if strokes and strokes[0]:
                prompt_options.append(("scribble", strokes[0]))

        if use_anchor_boxes and not ref.get("spatial_destination"):
            selected_prompt = (
                ("box", ref["bbox_xyxy"]) if ref["bbox_xyxy"] is not None else None
            )
        else:
            selected_prompt = rng.choice(prompt_options) if prompt_options else None

        if selected_prompt is not None:
            prompt_type, value = selected_prompt
            variable = f"{prompt_type}_{index}"
            ref["selected_prompt"] = {
                "type": prompt_type,
                "value": value,
                "variable": variable,
            }
            ref["display_token"] = f"<{variable}>"
            token = ref.get("token")
            if token:
                display_tokens[token] = ref["display_token"]
        else:
            ref["selected_prompt"] = None
            ref["display_token"] = ref.get("token")

    for token, display_token in display_tokens.items():
        instruction = instruction.replace(token, display_token)
    sample["instruction"] = instruction


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
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=color
                )
            elif points:
                draw.line(points, fill=(0, 220, 255, 255), width=line_width * 2)
                label_origin = points[0]
        label = ref.get("display_token") or ref.get("token") or f"reference_{index}"
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
                "data": json_safe(value.model_dump(), asset_dir, prefix, seen),
            }
        if hasattr(value, "to_dict"):
            return {
                "__type__": type(value).__name__,
                "data": json_safe(value.to_dict(), asset_dir, prefix, seen),
            }
        try:
            attributes = vars(value)
        except TypeError:
            pass
        except ModuleNotFoundError as exc:
            if exc.name != "_gdbm":
                raise
            return {
                "__type__": type(value).__name__,
                "__repr__": "Object attributes unavailable because _gdbm is missing",
            }
        else:
            return {
                "__type__": type(value).__name__,
                "data": json_safe(attributes, asset_dir, prefix, seen),
            }
        return {"__type__": type(value).__name__, "__repr__": repr(value)}
    finally:
        seen.discard(value_id)


def _write_raw_agent_trajectory(
    path: Path, raw_steps: list[dict[str, Any]], asset_dir: Path
) -> None:
    serialized_steps = json_safe(raw_steps, asset_dir, "agent_raw_trajectory")
    path.write_text(
        json.dumps(serialized_steps, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_or_generate_first_turn(
    agent: Any,
    first_turn: FirstTurnConfig,
    *,
    sample_id: str,
    prompt_mode: str,
    prompt_signature: str,
    task: str,
    objects: dict[str, Any],
    model: Any,
    entry: dict[str, Any],
) -> FirstTurnCommand:
    """Return the saved first-turn command, or generate and save a new one.

    Entries are keyed by prompt signature, so each prompt-type combination has its own command.
    """
    from framework.core.agent import FirstTurnCommand

    fingerprint = task_fingerprint(task, objects)
    if first_turn.mode == "replay":
        cached = first_turn.store.get(sample_id, prompt_mode, prompt_signature)
        if cached is not None and cached.fingerprint == fingerprint:
            entry["first_turn_source"] = "cache"
            return FirstTurnCommand(
                code=cached.code,
                thought=cached.thought,
                raw_output=cached.raw_output,
            )
        if cached is not None:
            tqdm.write(
                f"First-turn cache is stale for {sample_id}; regenerating.\n"
                f"  cached:  {cached.task!r} objects={cached.objects}\n"
                f"  current: {task!r} objects={sorted(objects)}"
            )
    command = agent.generate_first_turn(task, objects=objects)
    first_turn.store.put(
        FirstTurnEntry(
            sample_id=sample_id,
            prompt_mode=prompt_mode,
            prompt_signature=prompt_signature,
            fingerprint=fingerprint,
            task=task,
            objects=sorted(objects),
            thought=command.thought,
            code=command.code,
            raw_output=command.raw_output,
            model_id=getattr(model, "model_id", None),
        )
    )
    entry["first_turn_source"] = "generated"
    return command


def _file_stem(sample_id: str, annotation_index: int) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
    return f"{annotation_index:06d}_{safe_id}"


def load_saved_actions(run_dir: Path, file_stem: str) -> list[str]:
    """Return the code of every act step in a saved sample's trajectory.json, in order."""
    trajectory_path = run_dir / "logs" / file_stem / "trajectory.json"
    steps = json.loads(trajectory_path.read_text(encoding="utf-8"))
    return [step["act"] for step in steps if "act" in step]


def _copy_saved_trajectory(run_dir: Path, file_stem: str, log_dir: Path) -> None:
    source_dir = run_dir / "logs" / file_stem
    for name in ("trajectory.json", "trajectory.txt", "agent_raw_trajectory.json", "raw.json"):
        if (source_dir / name).is_file():
            shutil.copyfile(source_dir / name, log_dir / name)


def run_one(
    record: dict[str, Any],
    annotation_index: int,
    *,
    benchmark: str,
    prompt_mode: PromptMode,
    output_dir: Path,
    image_loader: HFImageLoader | MiraBenchImageLoader,
    model: Any,
    tools: list[Any],
    max_steps: int = 16,
    first_turn: FirstTurnConfig | None = None,
    from_run: Path | None = None,
) -> dict[str, Any]:
    _validate_prompt_mode(prompt_mode)
    from framework.core import Agent, local_executor

    seed = seed_random_generators()
    start = time.perf_counter()
    sample_id = record.get("sample_id") or f"annotation_{annotation_index}"
    file_stem = _file_stem(sample_id, annotation_index)
    log_dir = output_dir / "logs" / file_stem
    images_dir = output_dir / "results" / "images"
    log_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "benchmark": benchmark,
        "prompt_mode": prompt_mode,
        "sample_id": sample_id,
        "annotation_index": annotation_index,
        "instruction": _get_instruction(record, prompt_mode),
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
            prompt_mode=prompt_mode,
        )
        if not sample["instruction"]:
            raise ValueError(
                f"Record has no supported instruction field for {benchmark}."
            )
        sample["sample_id"] = sample.get("sample_id") or sample_id
        entry["target_available"] = sample["target_image"] is not None
        select_reference_prompts(sample, prompt_mode)
        task, objects = build_agent_task(sample)
        agent_context = build_agent_context(sample, prompt_mode)
        entry["instruction"] = task
        entry["reference_prompts"] = [
            {
                "token": ref.get("display_token") or ref["token"],
                "annotation_token": ref["token"],
                "variable": ref["selected_prompt"]["variable"],
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
            verbosity_level=0,
        )
        command = None
        agent_started_at = time.perf_counter()
        try:
            if from_run is not None:
                actions = load_saved_actions(from_run, file_stem)
                result = agent.replay_actions(actions, task, objects=objects)
                entry["replayed_actions"] = len(actions)
                entry["fallback_used"] = agent.fallback_used
            elif first_turn is None:
                result = agent.send(task, objects=objects)
            else:
                command = _load_or_generate_first_turn(
                    agent,
                    first_turn,
                    sample_id=str(sample_id),
                    prompt_mode=prompt_mode,
                    prompt_signature=prompt_signature(
                        ref["selected_prompt"]["type"]
                        for ref in sample["references"]
                        if ref.get("selected_prompt") is not None
                    ),
                    task=task,
                    objects=objects,
                    model=model,
                    entry=entry,
                )
                if first_turn.mode == "replay":
                    result = agent.replay_first_turn(command, task, objects=objects)
        finally:
            entry["agent_elapsed_seconds"] = round(
                time.perf_counter() - agent_started_at, 2
            )
        if result is None:
            # Generate mode: the first action is saved, not executed, so there is no output image.
            entry.update(
                {
                    "status": "first_turn_generated",
                    "first_turn_thought": command.thought,
                    "first_turn_code": command.code,
                }
            )
            return entry
        images = [obj for obj in result.message.objects if isinstance(obj, Image.Image)]
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
    finally:
        entry["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        trajectory_text_path = log_dir / "trajectory.txt"
        trajectory_json_path = log_dir / "trajectory.json"
        raw_trajectory_path = log_dir / "agent_raw_trajectory.json"
        variables_json_path = log_dir / "variables.json"
        raw_json_path = log_dir / "raw.json"
        entry.update(
            {
                "trajectory_text_path": str(trajectory_text_path),
                "trajectory_json_path": str(trajectory_json_path),
                "raw_trajectory_path": str(raw_trajectory_path),
                "variables_path": str(variables_json_path),
                "raw_path": str(raw_json_path),
            }
        )
        try:
            trajectory = agent.trajectory if agent is not None else []
            trajectory_text = agent.trajectory_text if agent is not None else ""
            raw_steps = agent.raw_steps if agent is not None else []
            agent_timings = (
                agent.timings
                if agent is not None
                else {
                    "prompt_builds": [],
                    "llm_calls": [],
                    "tool_calls": [],
                }
            )
            agent_timing_totals = {
                "prompt_build_seconds": round(
                    sum(
                        call["duration_seconds"]
                        for call in agent_timings["prompt_builds"]
                    ),
                    4,
                ),
                "llm_call_seconds": round(
                    sum(
                        call["duration_seconds"] for call in agent_timings["llm_calls"]
                    ),
                    4,
                ),
                "tool_call_seconds": round(
                    sum(
                        call["duration_seconds"] for call in agent_timings["tool_calls"]
                    ),
                    4,
                ),
            }
            variables = agent.variables if agent is not None else {}
            entry["agent_timings"] = agent_timings
            entry["agent_timing_totals"] = agent_timing_totals
            trajectory_text_path.write_text(trajectory_text, encoding="utf-8")
            assets_dir = log_dir / "debug_assets"
            serialized_variables = json_safe(variables, assets_dir, "variables")
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
            _write_raw_agent_trajectory(raw_trajectory_path, raw_steps, assets_dir)
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
                "elapsed_seconds": entry["elapsed_seconds"],
                "agent_elapsed_seconds": entry.get("agent_elapsed_seconds"),
                "agent_trajectory": trajectory,
                "agent_raw_steps": raw_steps,
                "agent_timings": agent_timings,
                "agent_timing_totals": agent_timing_totals,
                "raw_trajectory_path": str(raw_trajectory_path),
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
            tqdm.write(
                f"Failed to save trajectory/raw debug files for "
                f"{sample_id}: {save_error!r}"
            )
    if from_run is not None:
        _copy_saved_trajectory(from_run, file_stem, log_dir)
    return entry


def _clear_sample_artifacts(
    output_dir: Path, record: dict[str, Any], index: int
) -> Path:
    sample_id = record.get("sample_id") or f"annotation_{index}"
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
    file_stem = f"{index:06d}_{safe_id}"
    log_dir = output_dir / "logs" / file_stem
    images_dir = output_dir / "results" / "images"

    if log_dir.is_symlink():
        log_dir.unlink()
    elif log_dir.exists():
        shutil.rmtree(log_dir)

    for suffix in ("source", "annotated", "output", "target"):
        (images_dir / f"{file_stem}_{suffix}.png").unlink(missing_ok=True)
    return log_dir


def run_benchmark(
    records: list[dict[str, Any]],
    sample_indices: Iterable[int],
    *,
    benchmark: str,
    prompt_mode: PromptMode = "mixed",
    output_dir: Path,
    image_loader: HFImageLoader | MiraBenchImageLoader,
    model: Any,
    tools: list[Any],
    resume: bool = False,
    rerun_completed: bool = False,
    display_fn: Any = None,
    first_turn: FirstTurnConfig | None = None,
    from_run: Path | None = None,
) -> list[dict[str, Any]]:
    _validate_prompt_mode(prompt_mode)
    if rerun_completed and not resume:
        raise ValueError("rerun_completed requires resume=True.")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results" / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    selected_indices = list(dict.fromkeys(sample_indices))
    for annotation_index in selected_indices:
        if not 0 <= annotation_index < len(records):
            raise IndexError(f"Annotation index out of range: {annotation_index}")

    results_by_index: dict[int, dict[str, Any]] = {}
    if resume:
        if not results_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume; inference results not found: {results_path}"
            )
        positions_by_sample_id: dict[str, int] = {}
        for position, record in enumerate(records):
            sample_id = record.get("sample_id") or f"annotation_{position}"
            if sample_id in positions_by_sample_id:
                raise ValueError(f"Duplicate sample_id in annotations: {sample_id!r}")
            positions_by_sample_id[sample_id] = position

        for previous_result in read_jsonl(results_path):
            stored_index = previous_result.get("annotation_index")
            if not isinstance(stored_index, int) or isinstance(stored_index, bool):
                raise ValueError(
                    f"Invalid annotation_index in {results_path}: {stored_index!r}"
                )
            sample_id = previous_result.get("sample_id")
            if isinstance(sample_id, str):
                annotation_index = positions_by_sample_id.get(sample_id)
            elif 0 <= stored_index < len(records):
                annotation_index = stored_index
            else:
                annotation_index = None
            if annotation_index is None:
                raise ValueError(
                    f"Cannot match result annotation_index {stored_index!r} "
                    f"and sample_id {sample_id!r} to the annotation file."
                )
            if previous_result.get("benchmark") != benchmark:
                raise ValueError(
                    f"Cannot resume {benchmark!r} from results for "
                    f"{previous_result.get('benchmark')!r}."
                )
            if previous_result.get("prompt_mode") != prompt_mode:
                raise ValueError(
                    f"Cannot resume {prompt_mode!r} mode from results for "
                    f"{previous_result.get('prompt_mode')!r} mode."
                )
            expected_sample_id = records[annotation_index].get("sample_id") or (
                f"annotation_{annotation_index}"
            )
            metadata = records[annotation_index].get("metadata")
            row_index = (
                metadata.get("row_index") if isinstance(metadata, dict) else None
            )
            accepted_indices = {annotation_index}
            if isinstance(row_index, int) and not isinstance(row_index, bool):
                accepted_indices.add(row_index)
            if (
                previous_result.get("sample_id") != expected_sample_id
                or stored_index not in accepted_indices
            ):
                raise ValueError(
                    f"Cannot resume result at annotation index {stored_index}: "
                    f"expected sample {expected_sample_id!r} at position "
                    f"{annotation_index}, found "
                    f"{previous_result.get('sample_id')!r}."
                )
            previous_result["annotation_index"] = annotation_index
            if annotation_index in results_by_index:
                raise ValueError(
                    f"Duplicate result for annotation {annotation_index} in "
                    f"{results_path}."
                )
            results_by_index[annotation_index] = previous_result

    done_status = (
        "first_turn_generated"
        if first_turn is not None and first_turn.mode == "generate"
        else "completed"
    )
    pending_indices = [
        index
        for index in selected_indices
        if rerun_completed
        or results_by_index.get(index, {}).get("status") != done_status
    ]
    total = len(pending_indices)
    failures = []

    def persist_results() -> None:
        temporary_path = results_path.with_name(f".{results_path.name}.tmp")
        contents = "".join(
            json.dumps(result, ensure_ascii=False) + "\n"
            for _, result in sorted(results_by_index.items())
        )
        temporary_path.write_text(contents, encoding="utf-8")
        temporary_path.replace(results_path)

    if not resume and not pending_indices:
        persist_results()

    progress = tqdm(
        pending_indices,
        total=total,
        desc=f"{benchmark} inference",
        unit="sample",
    )
    for annotation_index in progress:
        sample_id = records[annotation_index].get(
            "sample_id", f"annotation_{annotation_index}"
        )
        progress.set_postfix(
            sample=str(sample_id),
            status="running",
            refresh=True,
        )
        if from_run is not None and not (
            from_run / "logs" / _file_stem(sample_id, annotation_index) / "trajectory.json"
        ).is_file():
            tqdm.write(f"Skipped {sample_id}: no saved trajectory in {from_run}.")
            results_by_index[annotation_index] = {
                "benchmark": benchmark,
                "prompt_mode": prompt_mode,
                "sample_id": sample_id,
                "annotation_index": annotation_index,
                "status": "skipped",
                "error": "No saved trajectory in the source run.",
            }
            persist_results()
            continue
        if rerun_completed:
            log_dir = _clear_sample_artifacts(
                output_dir, records[annotation_index], annotation_index
            )
            results_by_index[annotation_index] = {
                "benchmark": benchmark,
                "prompt_mode": prompt_mode,
                "sample_id": sample_id,
                "annotation_index": annotation_index,
                "status": "running",
                "log_dir": str(log_dir),
            }
            persist_results()
        result = run_one(
            records[annotation_index],
            annotation_index,
            benchmark=benchmark,
            prompt_mode=prompt_mode,
            output_dir=output_dir,
            image_loader=image_loader,
            model=model,
            tools=tools,
            first_turn=first_turn,
            from_run=from_run,
        )
        results_by_index[annotation_index] = result
        persist_results()
        progress.set_postfix(
            sample=str(result["sample_id"]),
            status=result["status"],
            refresh=False,
        )
        if result["status"] == "failed":
            failures.append(
                {
                    "sample_id": result["sample_id"],
                    "error": result.get("error", "Unknown error"),
                }
            )
        if result["status"] == "completed" and display_fn is not None:
            with Image.open(result["output_path"]) as output_image:
                display_fn(output_image.copy())
    for failure in failures:
        tqdm.write(f"Failed {failure['sample_id']}: {failure['error']}")
    if resume:
        persist_results()
    return [
        results_by_index[index]
        for index in selected_indices
        if index in results_by_index
    ]
