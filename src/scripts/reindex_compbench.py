#!/usr/bin/env python3
"""Reindex CompBench logs/images/results from annotations[*].metadata.row_index.

Safe by default: previews only. Use --apply after examining the plan.
Never identifies samples by old numeric indices.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LOG_RE = re.compile(r"^(?P<index>\d+)_(?P<slug>.+\.png)$")
# Based on observed files; unknown formats are rejected rather than silently skipped.
IMAGE_RE = re.compile(
    r"^(?P<index>\d+)_(?P<slug>.+\.png)_(?P<kind>annotated|output|source|target)\.png$"
)


def fail(message):
    raise ValueError(message)


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                fail(f"{path}:{number}: blank JSONL line")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{number}: malformed JSON: {exc}")
            if not isinstance(obj, dict):
                fail(f"{path}:{number}: expected a JSON object")
            records.append(obj)
    if not records:
        fail(f"{path}: empty file")
    return records


def get_sample_id(obj, context):
    sample_id = obj.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.endswith(".png"):
        fail(f"{context}: missing/invalid sample_id: {sample_id!r}")
    # Supported examples:
    #   add:001ca3cb/0/00003.png
    #   multi_object_add:02b72b8d_00002.png
    #   multi_object_add:03c21af7/00020.png
    # Different operations use different separators after the colon.
    # Validate both forms without requiring a slash.
    if sample_id.count(":") != 1:
        fail(f"{context}: unexpected sample_id format: {sample_id!r}")
    operation, relative_name = sample_id.split(":", 1)
    if (not re.fullmatch(r"[A-Za-z0-9_]+", operation)
            or not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\.png", relative_name)):
        fail(f"{context}: unexpected sample_id format: {sample_id!r}")
    return sample_id


def slug_of(sample_id):
    return sample_id.replace(":", "_").replace("/", "_")


def ensure_unique(mapping, key, context):
    if key in mapping:
        fail(f"Duplicate {key!r}: {mapping[key]} and {context}")


def require_directory(path, title):
    if path.is_symlink() or not path.is_dir():
        fail(f"{title} must be a real directory: {path}")


def require_file(path, title):
    if path.is_symlink() or not path.is_file():
        fail(f"{title} must be a regular file: {path}")


def plan_changes(annotations_path, logs_path, images_path, results_path, sort_results):
    require_file(annotations_path, "Annotations")
    require_file(results_path, "Results JSONL")
    require_directory(logs_path, "Logs")
    require_directory(images_path, "Images")
    if annotations_path.resolve() == results_path.resolve():
        fail("Annotations and results must be different files")
    if logs_path.resolve() == images_path.resolve():
        fail("Logs and images must be different directories")

    # 1. Ground truth: metadata.row_index, NOT annotation line number.
    ground_truth = {}
    slug_to_id = {}
    used_indices = {}
    for line_num, obj in enumerate(read_jsonl(annotations_path), 1):
        sample_id = get_sample_id(obj, f"annotations line {line_num}")
        metadata = obj.get("metadata")
        if not isinstance(metadata, dict):
            fail(f"annotations line {line_num}: missing metadata object")
        index = metadata.get("row_index")
        if type(index) is not int or index < 0:
            fail(f"annotations line {line_num}: invalid metadata.row_index: {index!r}")
        ensure_unique(ground_truth, sample_id, f"annotations line {line_num}")
        ensure_unique(used_indices, index, f"annotations line {line_num}")
        slug = slug_of(sample_id)
        ensure_unique(slug_to_id, slug, f"annotations line {line_num}")
        ground_truth[sample_id] = index
        slug_to_id[slug] = sample_id
        used_indices[index] = sample_id

    width = max(6, len(str(max(used_indices))))
    changes = []  # (source, destination) pairs
    log_prefix = {}
    image_kinds = defaultdict(set)
    old_index_mismatches = []

    # 2. Rename log directories, matching the slug (never the old prefix).
    for entry in logs_path.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_symlink() or not entry.is_dir():
            fail(f"Unexpected non-directory entry in logs/: {entry}")
        match = LOG_RE.fullmatch(entry.name)
        if not match:
            fail(f"Unrecognized log directory name: {entry.name}")
        slug = match.group("slug")
        sample_id = slug_to_id.get(slug)
        if sample_id is None:
            fail(f"Log directory not in annotations: {entry.name}")
        ensure_unique(log_prefix, sample_id, entry)
        log_prefix[sample_id] = match.group("index")
        new_name = f"{ground_truth[sample_id]:0{width}d}_{slug}"
        if entry.name != new_name:
            changes.append((entry, entry.with_name(new_name)))

    if not log_prefix:
        fail("No log directories found")

    # 3. Reindex every supported image variant.
    image_count = 0
    for entry in images_path.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_symlink() or not entry.is_file():
            fail(f"Unexpected non-file entry in images/: {entry}")
        match = IMAGE_RE.fullmatch(entry.name)
        if not match:
            fail(f"Unrecognized image name (expected *_{{annotated,output,source,target}}.png): {entry.name}")
        slug, kind = match.group("slug"), match.group("kind")
        sample_id = slug_to_id.get(slug)
        if sample_id is None:
            fail(f"Image not in annotations: {entry.name}")
        if sample_id not in log_prefix:
            fail(f"Image has no matching logs folder: {entry.name}")
        if match.group("index") != log_prefix[sample_id]:
            fail(f"Old image/log indices disagree for {sample_id}: "
                 f"image={match.group('index')}, log={log_prefix[sample_id]}")
        if kind in image_kinds[sample_id]:
            fail(f"Duplicate image variant {kind!r} for {sample_id}")
        image_kinds[sample_id].add(kind)
        image_count += 1
        new_name = f"{ground_truth[sample_id]:0{width}d}_{slug}_{kind}.png"
        if entry.name != new_name:
            changes.append((entry, entry.with_name(new_name)))

    missing_images = set(log_prefix) - set(image_kinds)
    if missing_images:
        fail(f"{len(missing_images)} log samples have no image files. Examples: "
             f"{sorted(missing_images)[:5]}")

    # 4. Update results by sample_id (and optionally sort the records).
    results = read_jsonl(results_path)
    result_by_id = {}
    changed_result_count = 0
    for line_num, obj in enumerate(results, 1):
        sample_id = get_sample_id(obj, f"results line {line_num}")
        if sample_id not in ground_truth:
            fail(f"results line {line_num}: no matching annotation: {sample_id}")
        ensure_unique(result_by_id, sample_id, f"results line {line_num}")
        result_by_id[sample_id] = line_num
        old_index = obj.get("annotation_index")
        if type(old_index) is not int or old_index < 0:
            fail(f"results line {line_num}: invalid annotation_index: {old_index!r}")
        if sample_id in log_prefix and old_index != int(log_prefix[sample_id]):
            old_index_mismatches.append((sample_id, old_index, log_prefix[sample_id]))
        new_index = ground_truth[sample_id]
        if old_index != new_index:
            changed_result_count += 1
        obj["annotation_index"] = new_index

    if set(result_by_id) != set(log_prefix):
        missing_results = sorted(set(log_prefix) - set(result_by_id))
        extra_results = sorted(set(result_by_id) - set(log_prefix))
        fail(f"Log/results sample sets differ. Missing results ({len(missing_results)}): "
             f"{missing_results[:5]}; results without logs ({len(extra_results)}): {extra_results[:5]}")

    initial_result_order = [obj["sample_id"] for obj in results]
    if sort_results:
        results.sort(key=lambda obj: ground_truth[obj["sample_id"]])
    order_changed = initial_result_order != [obj["sample_id"] for obj in results]

    # 5. Ensure destinations are unique, and none would overwrite unmanaged files.
    sources = {src for src, _ in changes}
    destinations = {}
    for src, dest in changes:
        if dest in destinations:
            fail(f"Name collision: {src} and {destinations[dest]} both target {dest}")
        destinations[dest] = src
        if os.path.lexists(dest) and dest not in sources:
            fail(f"Destination already exists and is not being renamed: {dest}")

    # Protect against case-folding collisions on case-insensitive filesystems.
    for folder in (logs_path, images_path):
        planned = {src: dest for src, dest in changes if src.parent == folder}
        effective_names = []
        for entry in folder.iterdir():
            effective_names.append(planned.get(entry, entry).name.casefold())
        if len(set(effective_names)) != len(effective_names):
            fail(f"Case-insensitive name collision in {folder}")

    return {
        "changes": sorted(changes, key=lambda p: str(p[0])),
        "results": results,
        "result_count": len(results),
        "changed_result_count": changed_result_count,
        "order_changed": order_changed,
        "image_count": image_count,
        "log_count": len(log_prefix),
        "annotation_count": len(ground_truth),
        "kind_counts": dict(sorted(Counter(k for v in image_kinds.values() for k in v).items())),
        "old_index_mismatches": old_index_mismatches,
    }


def apply_changes(plan, annotations_path, logs_path, images_path, results_path, sort_results):
    # Back up results BEFORE editing. Manifest makes name changes auditable/reversible.
    token = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = results_path.parent / f"reindex_backup_{stamp}_{token[:8]}"
    backup_dir.mkdir(mode=0o700)
    backup_results = backup_dir / "results.jsonl.original"
    shutil.copy2(results_path, backup_results)

    manifest = {
        "annotation_source": str(annotations_path.resolve()),
        "logs_dir": str(logs_path.resolve()),
        "images_dir": str(images_path.resolve()),
        "results_file": str(results_path.resolve()),
        "results_backup": str(backup_results.resolve()),
        "sort_results": sort_results,
        "renames": [{"old": str(src.resolve()), "new": str(dest.absolute())}
                    for src, dest in plan["changes"]],
    }
    with (backup_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    # Stage modified results on the SAME filesystem for atomic replacement.
    fd, staged_path = tempfile.mkstemp(
        prefix=".results_reindex_", suffix=".tmp", dir=results_path.parent
    )
    staged_results = Path(staged_path)
    journal = []  # ordered successful (source, destination) rename operations
    results_committed = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            for obj in plan["results"]:
                output.write(json.dumps(obj, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

        # Phase 1: move ALL changing names to unique temporary names, so
        # permutations/swaps cannot clobber one another.
        staged_renames = []
        for number, (src, dest) in enumerate(plan["changes"]):
            temp = src.with_name(f".__reindex_{token}_{number}__")
            if os.path.lexists(temp):
                fail(f"Temporary name already exists: {temp}")
            if not os.path.lexists(src):
                fail(f"Source disappeared since preflight: {src}")
            os.rename(src, temp)
            journal.append((src, temp))
            staged_renames.append((temp, dest))

        # Phase 2: move staged names to final names.
        for temp, dest in staged_renames:
            if os.path.lexists(dest):
                fail(f"Destination appeared unexpectedly: {dest}")
            os.rename(temp, dest)
            journal.append((temp, dest))

        # Commit JSONL LAST. It stays untouched if any rename failed.
        os.replace(staged_results, results_path)
        results_committed = True

        # Full post-commit validation. If it fails, restore original results and
        # the original file/directory names using the rename journal.
        checked = plan_changes(annotations_path, logs_path, images_path,
                               results_path, sort_results=False)
        if checked["changes"] or checked["changed_result_count"]:
            fail("Post-commit validation found remaining incorrect indices/names")
        if sort_results:
            indices = [obj["annotation_index"] for obj in checked["results"]]
            if indices != sorted(indices):
                fail("Post-commit validation found unsorted results")
    except BaseException:
        if results_committed:
            try:
                # Restore through a temp file rather than copying over a file in place.
                restore_path = results_path.with_name(f".results_restore_{token}.tmp")
                shutil.copy2(backup_results, restore_path)
                os.replace(restore_path, results_path)
            except OSError as exc:
                print(f"RESULT RESTORE FAILED: backup at {backup_results}: {exc}",
                      file=sys.stderr)
        for src, dest in reversed(journal):
            try:
                os.rename(dest, src)
            except OSError as exc:
                print(f"ROLLBACK FAILED: {dest} -> {src}: {exc}", file=sys.stderr)
        raise
    finally:
        if staged_results.exists():
            staged_results.unlink()

    return backup_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Ground-truth annotations JSONL")
    parser.add_argument("--logs", type=Path, required=True, help="Directory of sample log directories")
    parser.add_argument("--images", type=Path, required=True, help="Directory of sample image files")
    parser.add_argument("--results", type=Path, required=True, help="Results JSONL to update in-place")
    parser.add_argument("--sort-results", action="store_true", help="Also order results lines by corrected annotation_index")
    parser.add_argument("--apply", action="store_true", help="Perform changes (without this, preview only)")
    args = parser.parse_args()

    paths = (args.annotations.expanduser().absolute(), args.logs.expanduser().absolute(),
             args.images.expanduser().absolute(), args.results.expanduser().absolute())
    try:
        plan = plan_changes(*paths, sort_results=args.sort_results)
        print(f"Ground-truth annotations : {plan['annotation_count']}")
        print(f"Log directories          : {plan['log_count']}")
        print(f"Image files              : {plan['image_count']} {plan['kind_counts']}")
        print(f"Results records          : {plan['result_count']}")
        print(f"Log/image renames        : {len(plan['changes'])}")
        print(f"Changed result indices   : {plan['changed_result_count']}")
        print(f"Results order changes    : {plan['order_changed']}")
        print(f"Old result-vs-log index mismatches: {len(plan['old_index_mismatches'])}")
        for sid, res, log in plan["old_index_mismatches"][:5]:
            print(f"  NOTE: {sid}: results old={res}, log old={log}")
        for src, dest in plan["changes"][:12]:
            print(f"  {src.name}  ->  {dest.name}")
        if len(plan["changes"]) > 12:
            print(f"  ... {len(plan['changes']) - 12} more renames")
        if not args.apply:
            print("\nDRY RUN: no files changed. Review the mapping, then rerun with --apply.")
            return
        if not plan["changes"] and not plan["changed_result_count"] and not plan["order_changed"]:
            print("\nALREADY SYNCHRONIZED: nothing changed.")
            return
        backup_dir = apply_changes(plan, *paths, sort_results=args.sort_results)
        print(f"\nSUCCESS. Original results + rename manifest: {backup_dir}")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
