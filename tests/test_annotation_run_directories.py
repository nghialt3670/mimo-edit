from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from annotation.workflows import (
    create_annotation_run_directory,
    resolve_annotation_run_directory,
)


class AnnotationRunDirectoryTests(unittest.TestCase):
    def test_resolves_latest_created_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = create_annotation_run_directory(root)
            second = create_annotation_run_directory(root)

            self.assertNotEqual(first, second)
            self.assertEqual(resolve_annotation_run_directory(root), second)
            self.assertEqual(
                resolve_annotation_run_directory(root, first.name),
                first,
            )

    def test_latest_requires_an_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                resolve_annotation_run_directory(temporary_directory)


if __name__ == "__main__":
    unittest.main()
