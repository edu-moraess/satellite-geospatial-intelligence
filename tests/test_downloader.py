"""
Tests: src.downloader (ensure_output_directory)

This is the guardrail behind the original
NotADirectoryError / FileExistsError bug on
data/raw/<SCENE_ID>. Requires rasterio to import the module
(same as the rest of the app), so this runs wherever the app
itself runs.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.downloader import ensure_output_directory


class TestEnsureOutputDirectory(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_directory_when_nothing_exists(self):
        target = self.tmp / "raw" / "SCENE_A"

        result = ensure_output_directory(target)

        self.assertTrue(result.is_dir())

    def test_idempotent_when_directory_already_exists(self):
        target = self.tmp / "raw" / "SCENE_B"
        target.mkdir(parents=True)

        result = ensure_output_directory(target)

        self.assertTrue(result.is_dir())

    def test_self_heals_when_a_stray_file_blocks_the_scene_dir(self):
        # This is the exact real-world bug: a previous crashed
        # download left a FILE where the scene directory
        # (data/raw/<SCENE_ID>) needs to be created.
        parent = self.tmp / "raw"
        parent.mkdir(parents=True)

        stray_file = parent / "SCENE_C"
        stray_file.write_text("leftover junk from a crash")

        result = ensure_output_directory(stray_file)

        # Directory now exists where the file used to be...
        self.assertTrue(result.is_dir())

        # ...and the original file was quarantined next to it,
        # not silently destroyed.
        quarantined = list(
            parent.glob("SCENE_C.corrupted.*")
        )
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(
            quarantined[0].read_text(),
            "leftover junk from a crash",
        )

    def test_conflict_on_parent_raises_clear_error(self):
        # A FILE sitting where a PARENT directory needs to be
        # is a more unusual, structural conflict - it must
        # raise a clear, specific error instead of silently
        # auto-healing (which could affect other scenes).
        parent_as_file = self.tmp / "raw"
        parent_as_file.write_text("this should be a directory")

        target = parent_as_file / "SCENE_D"

        with self.assertRaises(RuntimeError):
            ensure_output_directory(target)


if __name__ == "__main__":
    unittest.main()
