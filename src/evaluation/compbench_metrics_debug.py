import base64
import html
from io import BytesIO
from pathlib import Path
from typing import Any

from IPython.display import HTML, display
from PIL import Image


def _resolve_artifact(
    result: dict[str, Any],
    field: str,
    output_dir: Path,
    sample_output_path: Any,
) -> Path | None:
    if isinstance(sample_output_path, str) and sample_output_path:
        output_path = Path(sample_output_path).expanduser()
        if field == "trajectory_text_path":
            sample_dir = output_path.name.removesuffix("_output.png")
            derived_path = output_dir / "logs" / sample_dir / "trajectory.txt"
        else:
            suffixes = {
                "source_path": "_source.png",
                "annotated_path": "_annotated.png",
                "target_path": "_target.png",
                "output_path": "_output.png",
            }
            suffix = suffixes.get(field)
            derived_path = (
                output_dir / "results" / "images"
                / output_path.name.replace("_output.png", suffix)
                if suffix is not None
                else output_path
            )
        if derived_path.is_file():
            return derived_path

    for value in (sample_output_path, result.get(field)):
        if not isinstance(value, str) or not value:
            continue
        recorded_path = Path(value).expanduser()
        if recorded_path.is_file():
            return recorded_path
        if field == "trajectory_text_path":
            local_path = output_dir / "logs" / recorded_path.parent.name / recorded_path.name
        else:
            local_path = output_dir / "results" / "images" / recorded_path.name
        if local_path.is_file():
            return local_path
    return None


def _image_card(title: str, path: Path | None) -> str:
    escaped_title = html.escape(title)
    if path is None:
        return (
            '<div><b>'
            f"{escaped_title}</b><p><i>Not available in this run</i></p></div>"
        )
    with Image.open(path) as opened_image:
        preview = opened_image.convert("RGB")
        preview.thumbnail((360, 360))
        buffer = BytesIO()
        preview.save(buffer, format="JPEG", quality=78, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    link = html.escape(path.resolve().as_uri(), quote=True)
    return (
        f"<div><b>{escaped_title}</b><a href=\"{link}\">"
        f"<img src=\"data:image/jpeg;base64,{encoded}\" alt=\"{escaped_title}\" "
        "style=\"display:block;max-width:100%;max-height:280px;object-fit:contain;margin-top:4px\"></a></div>"
    )


def _annotation_text(annotation: dict[str, Any]) -> str:
    items = []
    for reference in annotation.get("references") or []:
        region = reference.get("region") or {}
        parts = [reference.get(key) for key in ("token", "role", "span")]
        if region.get("label"):
            parts.append(region["label"])
        if region.get("bbox_xyxy") is not None:
            parts.append(f"bbox_xyxy={region['bbox_xyxy']}")
        description = html.escape(" | ".join(str(part) for part in parts if part))
        items.append(f"<li>{description}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else "No object references recorded."


def display_worst_sample_details(
    row: dict[str, Any],
    rank: int,
    metric: str,
    *,
    annotation: dict[str, Any],
    result: dict[str, Any],
    output_dir: Path,
    sample_output_path: str | None,
) -> None:
    sample_id = row["sample_id"]
    images = (
        (
            "Input",
            _resolve_artifact(result, "source_path", output_dir, sample_output_path),
        ),
        (
            "Input + annotation",
            _resolve_artifact(result, "annotated_path", output_dir, sample_output_path),
        ),
        (
            "Target/reference output",
            _resolve_artifact(result, "target_path", output_dir, sample_output_path),
        ),
        (
            "Generated output",
            _resolve_artifact(result, "output_path", output_dir, sample_output_path),
        ),
    )
    image_cards = "".join(_image_card(title, path) for title, path in images)
    trajectory_path = _resolve_artifact(
        result, "trajectory_text_path", output_dir, sample_output_path
    )
    if trajectory_path is None:
        trajectory = "<p><i>Trajectory not available in this run.</i></p>"
    else:
        text = trajectory_path.read_text(encoding="utf-8")
        trajectory = (
            f"<details><summary>Agent trajectory ({len(text.splitlines())} lines)</summary>"
            f"<pre style=\"white-space:pre-wrap;max-height:600px;overflow:auto\">"
            f"{html.escape(text)}</pre></details>"
        )
    task = html.escape(str(row.get("task") or ""))
    escaped_id = html.escape(str(sample_id))
    original = html.escape(str(annotation.get("original_instruction") or ""))
    mixed = html.escape(str(annotation.get("mixed_instruction") or ""))
    display(
        HTML(
            '<section style="border:1px solid #bbb;border-radius:6px;padding:12px;margin:12px 0">'
            f"<h4>Rank {rank} - {escaped_id} - {metric}: {row[metric]:.4f}</h4>"
            f"<p><b>Task:</b> {task}</p>"
            f"<p><b>Original instruction:</b> {original}</p>"
            f"<p><b>Mixed instruction:</b> {mixed}</p>"
            f"<b>Annotation references:</b>{_annotation_text(annotation)}"
            f'<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px">{image_cards}</div>'
            f"{trajectory}</section>"
        )
    )
