#!/usr/bin/env python3
"""Read-only consistency check for CompBench annotations, logs, images, results.

The source of truth is annotations.jsonl: sample_id -> metadata.row_index.
Does not rename, rewrite, or delete any files.

Examples of supported IDs and names:
  sample_id: multi_object_add:02b72b8d_00002.png
  logs/:    002297_multi_object_add_02b72b8d_00002.png/
  images/:  002297_multi_object_add_02b72b8d_00002.png_output.png
  results:  {"sample_id": "...", "annotation_index": 2297, ...}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LOG_RE = re.compile(r"^(?P<index>\d+)_(?P<slug>.+\.png)$")
IMAGE_RE = re.compile(
    r"^(?P<index>\d+)_(?P<slug>.+\.png)_(?P<variant>[a-zA-Z0-9_-]+)\.png$"
)


def slug_of(sample_id: str) -> str:
    return sample_id.replace(":", "_").replace("/", "_")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def valid_index(value) -> bool:
    return type(value) is int and value >= 0  # A bool is NOT an index.


class Report:
    def __init__(self, limit: int):
        self.limit = limit
        self.problems: dict[str, dict[str, list[str] | int]] = {}

    def add(self, category: str, message: str) -> None:
        entry = self.problems.setdefault(category, {"count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < self.limit:
            entry["examples"].append(message)

    def missing(self, category: str, missing_set, description) -> None:
        values = sorted(missing_set)
        if values:
            self.add(category, f"{len(values)} missing; first {min(len(values), self.limit)}: "
                                f"{[description(value) for value in values[:self.limit]]}")

    def display(self):
        if not self.problems:
            print("\nPASS: No duplicate indices, missing entries, or mismatched sample/index pairs.")
            return True
        print(f"\nFAIL: {len(self.problems)} issue categories detected:")
        for category, record in self.problems.items():
            print(f"\n  [{category}] ({record['count']} reported issues)")
            for message in record["examples"]:
                print(f"    - {message}")
        return False


def run(args):
    report = Report(args.show)
    annotations = args.annotations
    results = args.results
    logs_dir = args.logs
    images_dir = args.images

    for file_path in (annotations, results):
        if not file_path.is_file():
            raise ValueError(f"File does not exist: {file_path}")
    for folder in (logs_dir, images_dir):
        if not folder.is_dir():
            raise ValueError(f"Directory does not exist: {folder}")

    # Build source-of-truth maps. Keep both mappings, because index and
    # sample_id must each be unique.
    expected_by_idx = {}
    expected_by_id = {}
    expected_by_slug = {}
    n_annotations = 0
    for line_no, row in read_jsonl(annotations):
        n_annotations += 1
        sample_id = row.get("sample_id")
        metadata = row.get("metadata")
        index = metadata.get("row_index") if isinstance(metadata, dict) else None
        if not isinstance(sample_id, str) or sample_id.count(":") != 1 or not sample_id.endswith(".png"):
            report.add("annotation invalid sample_id", f"line {line_no}: {sample_id!r}")
            continue
        if not valid_index(index):
            report.add("annotation invalid row_index", f"line {line_no}: {index!r} for {sample_id}")
            continue
        if sample_id in expected_by_id:
            report.add("annotation duplicate sample_id", f"line {line_no}: {sample_id}")
        else:
            expected_by_id[sample_id] = index
        if index in expected_by_idx:
            report.add("annotation duplicate row_index", f"index {index}: {expected_by_idx[index]} AND {sample_id}")
        else:
            expected_by_idx[index] = sample_id
        slug = slug_of(sample_id)
        if slug in expected_by_slug and expected_by_slug[slug] != (index, sample_id):
            report.add("annotation duplicate normalized slug", f"slug {slug}: {expected_by_slug[slug]} AND {(index, sample_id)}")
        else:
            expected_by_slug[slug] = (index, sample_id)

    if not expected_by_idx:
        raise ValueError("No usable annotation entries found")
    expected_indices = set(expected_by_idx)
    expected_slugs = set(expected_by_slug)
    expected_count = len(expected_by_id)
    # row_index values are authoritative, and may be sparse because a dataset
    # was filtered or assembled from several subsets. Gaps are INFORMATIONAL
    # unless --require-contiguous is explicitly requested.
    max_index = max(expected_indices)
    min_index = min(expected_indices)
    unused_indices = set(range(min_index, max_index + 1)) - expected_indices
    if args.require_contiguous and expected_indices != set(range(expected_count)):
        absent = sorted(set(range(expected_count)) - expected_indices)
        extra = sorted(expected_indices - set(range(expected_count)))
        report.add("annotation index range", f"Expected 0..{expected_count - 1}; "
                   f"missing={absent[:args.show]}, outside={extra[:args.show]}")

    print("ANNOTATIONS (source of truth)")
    print(f"  lines: {n_annotations}; unique sample_ids: {expected_count}; "
          f"distinct row_index values: {len(expected_indices)}; index range: {min_index}..{max_index}")
    print(f"  unused indices within range: {len(unused_indices)} "
          f"({'NOT allowed by --require-contiguous' if args.require_contiguous else 'allowed'})")
    if unused_indices:
        print(f"  unused index examples: {sorted(unused_indices)[:args.show]}")

    def check_sample(section, index, slug, filename):
        matched = expected_by_slug.get(slug)
        if matched is None:
            report.add(f"{section}: unknown sample", filename)
        elif index != matched[0]:
            report.add(f"{section}: wrong index", f"{filename} -> expected index {matched[0]} ({matched[1]})")
        if index not in expected_by_idx:
            report.add(f"{section}: index outside annotations", f"{index} ({filename})")
        elif slug != slug_of(expected_by_idx[index]):
            # The above wrong-index check explains most of these; this message
            # makes index collisions particularly easy to investigate.
            report.add(f"{section}: index points to different sample", f"index {index} belongs to {expected_by_idx[index]}, not {filename}")

    # Logs should have exactly one directory per annotated sample/index.
    log_entries = defaultdict(list)
    log_sample_names = defaultdict(list)
    num_logs = 0
    for entry in logs_dir.iterdir():
        if entry.name.startswith("."):
            continue
        if not entry.is_dir() or entry.is_symlink():
            report.add("logs: unexpected entry", str(entry.name))
            continue
        match = LOG_RE.fullmatch(entry.name)
        if match is None:
            report.add("logs: invalid name", entry.name)
            continue
        num_logs += 1
        index = int(match.group("index"))
        slug = match.group("slug")
        log_entries[index].append(entry.name)
        log_sample_names[slug].append(entry.name)
        check_sample("logs", index, slug, entry.name)
    for index, names in log_entries.items():
        if len(names) > 1:
            report.add("logs: DUPLICATE INDEX", f"index {index}: {names[:args.show]}")
    for slug, names in log_sample_names.items():
        if len(names) > 1:
            report.add("logs: duplicate sample", f"{slug}: {names[:args.show]}")
    report.missing("logs: missing indices", expected_indices - set(log_entries), str)
    report.missing("logs: missing samples", expected_slugs - set(log_sample_names), str)
    print("\nLOGS")
    print(f"  directories: {num_logs}; distinct indices: {len(log_entries)}; distinct samples: {len(log_sample_names)}")
    print(f"  duplicate indices: {sum(len(v) > 1 for v in log_entries.values())}")

    # Multiple image files PER index are correct; they represent distinct
    # variants of the same sample. Duplicate index here means multiple slugs
    # assigned to the same index, NOT source+target+output+annotated.
    image_sample_by_idx = defaultdict(set)
    image_entries = defaultdict(list)  # (idx, slug, variant) -> filenames
    image_variants = defaultdict(set)  # slug -> variants
    variant_counts = Counter()
    image_count = 0
    for entry in images_dir.iterdir():
        if entry.name.startswith("."):
            continue
        if not entry.is_file() or entry.is_symlink():
            report.add("images: unexpected entry", entry.name)
            continue
        match = IMAGE_RE.fullmatch(entry.name)
        if match is None:
            report.add("images: invalid name", entry.name)
            continue
        image_count += 1
        index = int(match.group("index"))
        slug = match.group("slug")
        variant = match.group("variant")
        image_sample_by_idx[index].add(slug)
        image_entries[index, slug, variant].append(entry.name)
        image_variants[slug].add(variant)
        variant_counts[variant] += 1
        check_sample("images", index, slug, entry.name)
    for index, slugs in image_sample_by_idx.items():
        if len(slugs) > 1:
            report.add("images: DUPLICATE INDEX assigned to distinct samples", f"index {index}: {sorted(slugs)[:args.show]}")
    for (index, slug, variant), names in image_entries.items():
        if len(names) > 1:
            report.add("images: duplicate variant", f"index {index} / {slug} / {variant}: {names[:args.show]}")
    report.missing("images: missing indices", expected_indices - set(image_sample_by_idx), str)
    report.missing("images: missing samples", expected_slugs - set(image_variants), str)

    expected_variants = set(args.expected_variants) if args.expected_variants else set(variant_counts)
    for slug in sorted(expected_slugs):
        if slug not in image_variants:
            continue  # Already listed under missing samples.
        missing_variants = expected_variants - image_variants[slug]
        if missing_variants:
            report.add("images: missing variants", f"{slug}: {sorted(missing_variants)}")
    print("\nIMAGES")
    print(f"  files: {image_count}; distinct indices: {len(image_sample_by_idx)}; distinct samples: {len(image_variants)}")
    print(f"  variants: {dict(sorted(variant_counts.items()))}")
    print(f"  expected variants per sample: {sorted(expected_variants)} "
          f"({'specified' if args.expected_variants else 'inferred from files'})")
    print(f"  indices used by different samples: {sum(len(s) > 1 for s in image_sample_by_idx.values())}")

    # Results JSONL should have one record per annotated sample/index.
    result_entries = defaultdict(list)
    result_samples = defaultdict(list)
    results_rows = 0
    for line_no, row in read_jsonl(results):
        results_rows += 1
        sample_id = row.get("sample_id")
        index = row.get("annotation_index")
        if not isinstance(sample_id, str):
            report.add("results: invalid sample_id", f"line {line_no}: {sample_id!r}")
            continue
        result_samples[sample_id].append(line_no)
        if not valid_index(index):
            report.add("results: invalid annotation_index", f"line {line_no}: {index!r} ({sample_id})")
            continue
        result_entries[index].append((line_no, sample_id))
        expected = expected_by_id.get(sample_id)
        if expected is None:
            report.add("results: unknown sample", f"line {line_no}: {sample_id}")
        elif index != expected:
            report.add("results: wrong annotation_index", f"line {line_no}: {sample_id} has {index}, expected {expected}")
        if index not in expected_by_idx:
            report.add("results: index outside annotations", f"line {line_no}: {index}")
        elif sample_id != expected_by_idx[index]:
            report.add("results: index points to different sample", f"line {line_no}: {index} belongs to {expected_by_idx[index]}, not {sample_id}")
    for index, rows in result_entries.items():
        if len(rows) > 1:
            report.add("results: DUPLICATE INDEX", f"index {index}: {rows[:args.show]}")
    for sample_id, lines in result_samples.items():
        if len(lines) > 1:
            report.add("results: duplicate sample_id", f"{sample_id}: lines {lines[:args.show]}")
    report.missing("results: missing indices", expected_indices - set(result_entries), str)
    report.missing("results: missing samples", set(expected_by_id) - set(result_samples), str)
    print("\nRESULTS JSONL")
    print(f"  rows: {results_rows}; distinct indices: {len(result_entries)}; distinct samples: {len(result_samples)}")
    print(f"  duplicate indices: {sum(len(v) > 1 for v in result_entries.values())}")
    return 0 if report.display() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True, help="Annotation JSONL (source of truth)")
    parser.add_argument("--logs", type=Path, required=True, help="Directory containing 000000_<sample>.png log folders")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing 000000_<sample>.png_<variant>.png files")
    parser.add_argument("--results", type=Path, required=True, help="Results JSONL with annotation_index")
    parser.add_argument("--expected-variants", nargs="+", help="Optional explicitly required variants, e.g. source output target annotated")
    parser.add_argument("--show", type=int, default=8, help="Maximum examples per issue category (default: 8)")
    parser.add_argument("--require-contiguous", action="store_true", help="Treat annotation index gaps as errors (off by default)")
    args = parser.parse_args()
    if args.show < 1:
        parser.error("--show must be at least 1")
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
