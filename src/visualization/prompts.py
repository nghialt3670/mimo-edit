from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _mask_array(
    reference: dict[str, Any], base_dir: Path
) -> tuple[np.ndarray, tuple[int, int], str]:
    region = reference.get("region", {})
    image_width, image_height = (
        int(value) for value in region.get("image_size", (0, 0))
    )
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Invalid image_size for {reference.get('token', 'reference')}"
        )

    mask_path = region.get("mask_path")
    if mask_path:
        path = Path(mask_path)
        path = path if path.is_absolute() else base_dir / path
        if path.is_file():
            with Image.open(path) as image:
                mask = image.convert("L")
                if mask.size != (image_width, image_height):
                    mask = mask.resize(
                        (image_width, image_height), Image.Resampling.NEAREST
                    )
                binary = np.asarray(mask) > 0
            if not binary.any():
                raise ValueError(
                    f"Empty mask for {reference.get('token', 'reference')}: {path}"
                )
            return binary, (image_width, image_height), "mask"

    box = region.get("bbox_xyxy")
    if not box or len(box) != 4:
        raise ValueError(
            f"No usable mask or bounding box for {reference.get('token', 'reference')}"
        )
    x0, y0, x1, y1 = (float(value) for value in box)
    left = max(0, min(image_width, math.floor(x0)))
    top = max(0, min(image_height, math.floor(y0)))
    right = max(0, min(image_width, math.ceil(x1)))
    bottom = max(0, min(image_height, math.ceil(y1)))
    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid bounding box for {reference.get('token', 'reference')}"
        )
    fallback = Image.new("L", (image_width, image_height), 0)
    ImageDraw.Draw(fallback).rectangle((left, top, right - 1, bottom - 1), fill=255)
    return np.asarray(fallback) > 0, (image_width, image_height), "bbox_approximation"


def _manhattan_distance_to_background(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = padded.shape
    x_coordinates = np.arange(width)
    horizontal = np.empty((height, width), dtype=np.int32)

    for y, row in enumerate(padded):
        background = ~row
        nearest_left = np.maximum.accumulate(
            np.where(background, x_coordinates, -width)
        )
        nearest_right = np.minimum.accumulate(
            np.where(background, x_coordinates, width * 2)[::-1]
        )[::-1]
        horizontal[y] = np.minimum(
            x_coordinates - nearest_left, nearest_right - x_coordinates
        )
        horizontal[y, background] = 0

    distance = horizontal
    for y in range(1, height):
        distance[y] = np.minimum(distance[y], distance[y - 1] + 1)
    for y in range(height - 2, -1, -1):
        distance[y] = np.minimum(distance[y], distance[y + 1] + 1)
    return distance[1:-1, 1:-1]


def _make_scribbles(mask: np.ndarray, seed: int) -> list[list[list[float]]]:
    from scipy.ndimage import gaussian_filter1d
    from scipy.sparse.csgraph import connected_components, dijkstra
    from skimage.graph import pixel_graph
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask)
    graph, pixel_indices = pixel_graph(skeleton, connectivity=2)
    if graph.shape[0] < 2:
        return []

    # Remove short endpoint-to-junction branches before choosing the main path.
    minimum_branch_length = max(4.0, math.hypot(*mask.shape) * 0.025)
    while graph.shape[0] > 2:
        degrees = np.diff(graph.indptr)
        branches_to_remove: set[int] = set()
        for endpoint in np.flatnonzero(degrees == 1):
            branch = [int(endpoint)]
            previous = -1
            current = int(endpoint)
            branch_length = 0.0
            while True:
                neighbors = graph.indices[
                    graph.indptr[current] : graph.indptr[current + 1]
                ]
                next_nodes = neighbors[neighbors != previous]
                if len(next_nodes) != 1:
                    break
                next_node = int(next_nodes[0])
                branch_length += float(graph[current, next_node])
                previous, current = current, next_node
                if degrees[current] != 2:
                    if degrees[current] >= 3 and branch_length < minimum_branch_length:
                        branches_to_remove.update(branch)
                    break
                branch.append(current)

        if not branches_to_remove:
            break
        keep = np.ones(graph.shape[0], dtype=bool)
        keep[list(branches_to_remove)] = False
        graph = graph[keep][:, keep].tocsr()
        pixel_indices = pixel_indices[keep]

    component_count, component_labels = connected_components(graph, directed=False)
    if component_count > 1:
        largest_component = int(np.argmax(np.bincount(component_labels)))
        keep = component_labels == largest_component
        graph = graph[keep][:, keep].tocsr()
        pixel_indices = pixel_indices[keep]

    degrees = np.diff(graph.indptr)
    endpoints = np.flatnonzero(degrees == 1)
    candidates = endpoints if len(endpoints) >= 2 else np.arange(graph.shape[0])
    first_distances = dijkstra(graph, directed=False, indices=int(candidates[0]))
    start = int(candidates[np.argmax(first_distances[candidates])])
    distances, predecessors = dijkstra(
        graph, directed=False, indices=start, return_predecessors=True
    )
    end = int(candidates[np.argmax(distances[candidates])])

    path = [end]
    while path[-1] != start:
        previous = int(predecessors[path[-1]])
        if previous < 0:
            return []
        path.append(previous)
    path.reverse()
    if len(path) < 2:
        return []

    rows, columns = np.unravel_index(pixel_indices[path], mask.shape)
    coordinates = np.column_stack((rows, columns)).astype(float)
    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative_length[-1])
    if total_length < 4:
        return []

    generator = random.Random(seed)
    inset = total_length * generator.uniform(0.04, 0.08)
    sample_distances = np.arange(inset, total_length - inset, 1.0)
    if len(sample_distances) < 2:
        return []
    sampled = np.column_stack(
        (
            np.interp(sample_distances, cumulative_length, coordinates[:, 0]),
            np.interp(sample_distances, cumulative_length, coordinates[:, 1]),
        )
    )

    for smoothing in (2.0, 1.0, 0.0):
        curve = (
            gaussian_filter1d(sampled, sigma=smoothing, axis=0, mode="nearest")
            if smoothing
            else sampled
        )
        sample_rows = np.clip(
            np.floor(curve[:, 0] + 0.5).astype(int), 0, mask.shape[0] - 1
        )
        sample_columns = np.clip(
            np.floor(curve[:, 1] + 0.5).astype(int), 0, mask.shape[1] - 1
        )
        if mask[sample_rows, sample_columns].all():
            stroke = [[float(column + 0.5), float(row + 0.5)] for row, column in curve]
            return [stroke]

    return []


def build_visual_prompts(
    reference: dict[str, Any],
    base_dir: str | Path,
    sample_id: str,
) -> dict[str, Any]:
    mask, (width, height), geometry_source = _mask_array(reference, Path(base_dir))
    ys, xs = np.where(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
    distance = _manhattan_distance_to_background(mask)
    point_y, point_x = np.unravel_index(int(distance.argmax()), distance.shape)
    seed_value = int.from_bytes(
        hashlib.sha256(f"{sample_id}:{reference.get('token', '')}".encode()).digest()[
            :8
        ],
        "big",
    )
    stroke_width = max(2, round(min(width, height) * 0.006))
    strokes = _make_scribbles(mask, seed_value)

    return {
        "geometry_source": geometry_source,
        "coordinate_space": "pixel",
        "image_size": [width, height],
        "box": {
            "xyxy": [x0, y0, x1, y1],
            "normalized_xyxy": [x0 / width, y0 / height, x1 / width, y1 / height],
        },
        "point": {
            "xy": [point_x + 0.5, point_y + 0.5],
            "normalized_xy": [(point_x + 0.5) / width, (point_y + 0.5) / height],
            "method": "maximum_manhattan_distance_to_mask_boundary",
        },
        "scribble": {
            "strokes": strokes,
            "normalized_strokes": [
                [[x / width, y / height] for x, y in stroke] for stroke in strokes
            ],
            "stroke_width_px": stroke_width,
            "seed": seed_value,
            "method": "seeded_pruned_skeleton_geodesic",
            "valid": bool(strokes),
        },
    }


def backfill_visual_prompts(
    path: str | Path, overwrite: bool = False
) -> dict[str, int]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc

    generated = 0
    existing = 0
    for record in records:
        for reference in record.get("references", []):
            visual_prompts = reference.get("visual_prompts", {})
            scribble = (
                visual_prompts.get("scribble", {})
                if isinstance(visual_prompts, dict)
                else {}
            )
            if (
                visual_prompts
                and scribble.get("method") == "seeded_pruned_skeleton_geodesic"
                and not overwrite
            ):
                existing += 1
                continue
            reference["visual_prompts"] = build_visual_prompts(
                reference, path.parent, str(record.get("sample_id", "sample"))
            )
            generated += 1

    if generated:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)

    return {
        "records": len(records),
        "generated": generated,
        "already_present": existing,
    }
