from __future__ import annotations

import atexit
from difflib import SequenceMatcher
import html
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image

WORD_RE = re.compile(r"\b[\w'-]+\b")


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
    return records


class HuggingFaceRowImageLoader:
    def __init__(self) -> None:
        self._dataset_key: tuple[str, str] | None = None
        self._iterator: Any = None
        self._next_index = 0
        atexit.register(self.close)

    def close(self) -> None:
        iterator = self._iterator
        if iterator is not None:
            close_iterator = getattr(iterator, "close", None)
            if close_iterator is not None:
                try:
                    close_iterator()
                except Exception:
                    pass
        self._dataset_key = None
        self._iterator = None
        self._next_index = 0

    @staticmethod
    def _parse_reference(reference: str | None) -> tuple[str, str, int, str] | None:
        if not reference or not reference.startswith("hf://"):
            return None
        parts = reference[5:].split("/")
        if len(parts) < 4:
            return None
        try:
            row_index = int(parts[-2])
        except ValueError:
            return None
        dataset_id = "/".join(parts[:-3])
        split, field = parts[-3], parts[-1]
        if not dataset_id:
            return None
        return dataset_id, split, row_index, field

    def _row_at(self, dataset_id: str, split: str, row_index: int) -> dict[str, Any]:
        from datasets import load_dataset

        key = (dataset_id, split)
        if self._dataset_key != key or row_index < self._next_index:
            self.close()
            dataset = load_dataset(dataset_id, split=split, streaming=True)
            self._dataset_key = key
            self._iterator = iter(dataset)
            self._next_index = 0
        row = None
        while self._next_index <= row_index:
            try:
                row = next(self._iterator)
            except StopIteration as exc:
                raise IndexError(f"Dataset row {row_index} is unavailable in {dataset_id}/{split}") from exc
            self._next_index += 1
        assert row is not None
        return row

    @staticmethod
    def _as_image(value: Any) -> Image.Image | None:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            image_path = value.get("path")
            if image_bytes:
                from io import BytesIO

                with Image.open(BytesIO(image_bytes)) as image:
                    return image.convert("RGB")
            if image_path and Path(image_path).is_file():
                with Image.open(image_path) as image:
                    return image.convert("RGB")
        return None

    def load(self, record: dict[str, Any]) -> tuple[Image.Image | None, Image.Image | None]:
        source_ref = self._parse_reference(record.get("source_ref"))
        target_ref = self._parse_reference(record.get("target_ref"))
        if source_ref is None:
            return None, None

        dataset_id, split, row_index, source_field = source_ref
        row = self._row_at(dataset_id, split, row_index)
        source = self._as_image(row.get(source_field))

        target = None
        if target_ref is not None:
            target_dataset, target_split, target_index, target_field = target_ref
            if (target_dataset, target_split, target_index) == (dataset_id, split, row_index):
                target = self._as_image(row.get(target_field))
            else:
                target_row = self._row_at(target_dataset, target_split, target_index)
                target = self._as_image(target_row.get(target_field))
        return source, target


def _load_asset(value: str | None, base_dir: Path) -> Image.Image | None:
    if not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else base_dir / path
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.copy()


def _highlight_original_instruction(record: dict[str, Any]) -> str:
    original = str(record.get("original_instruction", ""))
    references = record.get("references", [])
    parser_instruction = record.get("parser_instruction") or record.get("metadata", {}).get("edit_instruction_single")
    spans: list[tuple[int, int]] = []
    search_from = 0

    for reference in sorted(references, key=lambda item: item.get("start", 0)):
        phrase = str(reference.get("span", ""))
        start = original.find(phrase, search_from) if phrase else -1
        if start >= 0:
            end = start + len(phrase)
            spans.append((start, end))
            search_from = end

    if parser_instruction and parser_instruction != original:
        original_words = list(WORD_RE.finditer(original))
        parser_words = [match.group().casefold() for match in WORD_RE.finditer(parser_instruction)]
        original_word_values = [match.group().casefold() for match in original_words]
        matcher = SequenceMatcher(None, original_word_values, parser_words, autojunk=False)
        for operation, original_start, original_end, _, _ in matcher.get_opcodes():
            if operation == "delete" and original_start < original_end:
                spans.append((original_words[original_start].start(), original_words[original_end - 1].end()))
    else:
        for reference in references:
            start, end = reference.get("start"), reference.get("end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(original):
                if original[start:end] == reference.get("span"):
                    spans.append((start, end))

    merged_spans: list[list[int]] = []
    for start, end in sorted(spans):
        if merged_spans and start <= merged_spans[-1][1]:
            merged_spans[-1][1] = max(merged_spans[-1][1], end)
        else:
            merged_spans.append([start, end])

    pieces: list[str] = []
    cursor = 0
    for start, end in merged_spans:
        pieces.extend((html.escape(original[cursor:start]), f"<strong>{html.escape(original[start:end])}</strong>"))
        cursor = end
    pieces.append(html.escape(original[cursor:]))
    return "".join(pieces)


def _highlight_mixed_instruction(record: dict[str, Any]) -> str:
    instruction = html.escape(str(record.get("mixed_instruction", "")))
    for reference in record.get("references", []):
        token = html.escape(str(reference.get("token", "")))
        if token:
            instruction = instruction.replace(token, f"<strong>{token}</strong>")
    return instruction


def _display_details(record: dict[str, Any], source_error: str | None) -> None:
    from IPython.display import Markdown, display

    original_instruction = str(record.get("original_instruction", ""))
    lines = [
        f"### {record.get('benchmark', 'Benchmark')} / {record.get('sample_id', 'sample')}",
        f"**Split:** `{record.get('split', '')}`",
        "",
        "<div style=\"margin: 8px 0 4px; color: #171717; font-weight: 700;\">Original instruction</div>",
        f"<div style=\"background-color: #f1f3f4; color: #171717; padding: 10px 14px; border-left: 4px solid #00798c; line-height: 1.6;\">{_highlight_original_instruction(record)}</div>",
    ]
    lines.extend((
        "",
        "<div style=\"margin: 10px 0 4px; color: #171717; font-weight: 700;\">Mixed instruction</div>",
    ))
    lines.extend((
        f"<div style=\"background-color: #f1f3f4; color: #171717; padding: 10px 14px; border-left: 4px solid #d1495b; line-height: 1.6;\">{_highlight_mixed_instruction(record)}</div>",
        "",
        f"**Source:** `{record.get('source_ref')}`  ",
        f"**Target:** `{record.get('target_ref')}`",
    ))
    if source_error:
        lines.extend(("", f"> Source/target reload failed: `{source_error}`"))

    lines.extend(("", "#### References"))
    references = record.get("references", [])
    if not references:
        lines.append("No references recorded.")
    for reference in references:
        region = reference.get("region", {})
        lines.extend(
            (
                f"- **{reference.get('token')}** (`{reference.get('role')}`): \"{reference.get('span')}\"",
                f"  Grounding query: `{reference.get('grounding_query')}`; detected as `{region.get('label')}`",
                f"  Box: `{region.get('bbox_xyxy')}` / image `{region.get('image_size')}`; confidence `{region.get('confidence')}`; source `{region.get('source')}`",
            )
        )

    lines.extend(("", "#### Validation and metadata", "```json"))
    lines.append(json.dumps({
        "validation": record.get("validation", {}),
        "linguistic_stats": record.get("linguistic_stats", {}),
        "metadata": record.get("metadata", {}),
    }, ensure_ascii=False, indent=2))
    lines.append("```")
    display(Markdown("\n".join(lines)))


def show_annotation(
    record: dict[str, Any],
    base_dir: str | Path,
    image_loader: HuggingFaceRowImageLoader | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    base_dir = Path(base_dir)
    source = target = None
    source_error = None
    if image_loader is not None:
        try:
            source, target = image_loader.load(record)
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"

    references = record.get("references", [])
    figure, axes = plt.subplots(max(1, 1 + len(references)), 4, figsize=(21, 5 + 3.5 * len(references)), squeeze=False)
    figure.patch.set_facecolor("#f4f3ee")

    role_colors = {"target": "#d1495b", "anchor": "#00798c", "source": "#edae49", "context": "#30638e"}
    for column, variant in enumerate(("box", "point", "scribble")):
        prompt_axis = axes[0][column]
        if source is None:
            prompt_axis.text(0.5, 0.5, "Source image unavailable", ha="center", va="center")
        else:
            prompt_axis.imshow(source)
            for reference in references:
                region = reference.get("region", {})
                prompts = reference.get("visual_prompts", region.get("visual_prompts", {}))
                prompt = prompts.get(variant, {})
                color = role_colors.get(reference.get("role"), "#00798c")
                if variant == "box":
                    box = prompt.get("xyxy", region.get("bbox_xyxy"))
                    if box and len(box) == 4:
                        x0, y0, x1, y1 = box
                        prompt_axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
                        prompt_axis.text(x0, max(0, y0 - 4), reference.get("token", "ref"), color="white", fontsize=9, weight="bold", bbox={"facecolor": color, "edgecolor": "none", "pad": 2})
                elif variant == "point":
                    point = prompt.get("xy")
                    if point and len(point) == 2:
                        x, y = point
                        prompt_axis.scatter([x], [y], s=110, marker="o", facecolors="none", edgecolors=color, linewidths=2)
                        prompt_axis.scatter([x], [y], s=24, marker="o", color=color)
                        prompt_axis.text(x + 4, y, reference.get("token", "ref"), color="white", fontsize=9, weight="bold", bbox={"facecolor": color, "edgecolor": "none", "pad": 2})
                else:
                    scribble = prompt.get("strokes", [])
                    width = prompt.get("stroke_width_px", 3)
                    for stroke in scribble:
                        if len(stroke) >= 2:
                            prompt_axis.plot([point[0] for point in stroke], [point[1] for point in stroke], color=color, linewidth=width, solid_capstyle="round")
                    if scribble and scribble[0]:
                        x, y = scribble[0][0]
                        prompt_axis.text(x, y, reference.get("token", "ref"), color="white", fontsize=9, weight="bold", bbox={"facecolor": color, "edgecolor": "none", "pad": 2})
        prompt_axis.set_title({"box": "Box prompt", "point": "Point prompt", "scribble": "Scribble prompt"}[variant])
        prompt_axis.axis("off")

    target_axis = axes[0][3]
    if target is None:
        target_axis.text(0.5, 0.5, "No target image available", ha="center", va="center")
    else:
        target_axis.imshow(target)
    target_axis.set_title("Benchmark target")
    target_axis.axis("off")

    for index, reference in enumerate(references, 1):
        region = reference.get("region", {})
        crop = _load_asset(region.get("crop_path"), base_dir)
        mask = _load_asset(region.get("mask_path"), base_dir)
        crop_axis, mask_axis, detail_axis, geometry_axis = axes[index]
        if crop is None:
            crop_axis.text(0.5, 0.5, "Crop asset unavailable", ha="center", va="center")
        else:
            crop_axis.imshow(crop)
        crop_axis.set_title(f"{reference.get('token')} crop ({reference.get('role')})")
        crop_axis.axis("off")
        if mask is None:
            mask_axis.text(0.5, 0.5, "No saved mask", ha="center", va="center")
        else:
            mask_axis.imshow(mask, cmap="gray")
        mask_axis.set_title("Reference mask")
        mask_axis.axis("off")
        detail_axis.axis("off")
        detail_axis.text(
            0,
            1,
            "\n".join((
                f"Span: {reference.get('span')}",
                f"Role: {reference.get('role')}",
                f"Query: {reference.get('grounding_query')}",
                f"Detected: {region.get('label')}",
                f"Confidence: {region.get('confidence')}",
                f"Box: {region.get('bbox_xyxy')}",
                f"Box source: {region.get('source')}",
            )),
            va="top",
            wrap=True,
            fontsize=10,
        )
        geometry_axis.axis("off")
        prompts = region.get("visual_prompts", {})
        scribble = prompts.get("scribble", {})
        geometry_axis.text(
            0,
            1,
            "\n".join((
                f"Geometry source: {prompts.get('geometry_source', 'not generated')}",
                f"Box: {prompts.get('box', {}).get('xyxy', region.get('bbox_xyxy'))}",
                f"Point: {prompts.get('point', {}).get('xy', 'not generated')}",
                f"Scribble strokes: {len(scribble.get('strokes', []))}",
                f"Scribble seed: {scribble.get('seed', 'not generated')}",
            )),
            va="top",
            wrap=True,
            fontsize=10,
        )

    figure.tight_layout()
    _display_details(record, source_error)
    plt.show()


class AnnotationBrowser:
    def __init__(self, annotation_file: str | Path) -> None:
        self.annotation_file = Path(annotation_file)
        self.records = load_annotations(self.annotation_file)
        self.image_loader = HuggingFaceRowImageLoader()

    def show(self, index: int = 0) -> None:
        if not self.records:
            raise ValueError(f"No annotation records found in {self.annotation_file}")
        if index < 0 or index >= len(self.records):
            raise IndexError(f"Sample index {index} is outside 0..{len(self.records) - 1}")
        show_annotation(self.records[index], self.annotation_file.parent, self.image_loader)

    def interact(self) -> None:
        from IPython.display import clear_output, display
        import ipywidgets as widgets

        if not self.records:
            raise ValueError(f"No annotation records found in {self.annotation_file}")
        clear_output(wait=True)
        control = widgets.interactive(
            self.show,
            index=widgets.BoundedIntText(min=0, max=len(self.records) - 1, value=0, description="Sample", continuous_update=False),
        )
        display(control)