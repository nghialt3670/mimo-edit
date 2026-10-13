from __future__ import annotations

import unittest

from evaluation.compbench_selection import select_compbench_indices


def _records(*tasks: str) -> list[dict[str, object]]:
    return [{"metadata": {"task": task}} for task in tasks]


class CompBenchSelectionTests(unittest.TestCase):
    def test_defaults_to_all_original_indices(self):
        self.assertEqual(
            select_compbench_indices(_records("add", "remove", "replace")),
            [0, 1, 2],
        )

    def test_filters_by_task(self):
        self.assertEqual(
            select_compbench_indices(
                _records("add", "remove", "add"), tasks={"add"}
            ),
            [0, 2],
        )

    def test_start_index_uses_original_annotation_positions(self):
        self.assertEqual(
            select_compbench_indices(
                _records("add", "remove", "add", "remove"),
                tasks={"remove"},
                start_index=1,
            ),
            [1, 3],
        )

    def test_limit_applies_after_task_filtering(self):
        self.assertEqual(
            select_compbench_indices(
                _records("add", "remove", "add", "remove"),
                tasks={"remove"},
                limit=1,
            ),
            [1],
        )

    def test_rejects_empty_task_filter(self):
        with self.assertRaisesRegex(ValueError, "one or more"):
            select_compbench_indices(_records("add"), tasks=set())

    def test_rejects_unknown_task_with_available_task_names(self):
        with self.assertRaisesRegex(ValueError, "Available tasks: add, remove"):
            select_compbench_indices(
                _records("add", "remove"), tasks={"replace"}
            )

    def test_rejects_start_index_outside_nonempty_annotations(self):
        with self.assertRaisesRegex(ValueError, "outside the annotation file"):
            select_compbench_indices(_records("add"), start_index=1)

    def test_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            select_compbench_indices(_records("add"), limit=0)


if __name__ == "__main__":
    unittest.main()
