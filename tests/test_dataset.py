import tempfile
import unittest
from pathlib import Path

from minisynth.dataset import patch_filename, write_random_patch_files
from minisynth.io import load_patch


class TestDatasetGeneration(unittest.TestCase):
    def test_patch_filename_includes_index_and_seed(self):
        self.assertEqual(patch_filename(3, 42), "patch_000003_seed_42.json")

    def test_write_random_patch_files_saves_json_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_random_patch_files(tmpdir, seed=10, count=3)

            self.assertEqual(len(paths), 3)
            self.assertEqual(paths[0], Path(tmpdir) / "patch_000000_seed_10.json")
            self.assertEqual(paths[1], Path(tmpdir) / "patch_000001_seed_11.json")
            self.assertEqual(paths[2], Path(tmpdir) / "patch_000002_seed_12.json")

            for path in paths:
                self.assertTrue(path.exists())
                self.assertIn("osc1_wave", load_patch(path))

    def test_write_random_patch_files_rejects_invalid_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_random_patch_files(tmpdir, count=0)


if __name__ == "__main__":
    unittest.main()
