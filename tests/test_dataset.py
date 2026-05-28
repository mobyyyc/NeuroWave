import tempfile
import unittest
from pathlib import Path

import soundfile as sf

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.dataset import (
    audio_filename,
    patch_filename,
    write_random_dataset_files,
    write_random_patch_files,
)
from minisynth.io import load_patch


class TestDatasetGeneration(unittest.TestCase):
    def test_patch_filename_includes_index_and_seed(self):
        self.assertEqual(patch_filename(3, 42), "patch_000003_seed_42.json")

    def test_audio_filename_includes_index_and_seed(self):
        self.assertEqual(audio_filename(3, 42), "patch_000003_seed_42.wav")

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

    def test_write_random_dataset_files_saves_params_and_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            records = write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                seed=30,
                count=2,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["seed"], 30)
            self.assertEqual(
                records[0]["patch_path"],
                root / "params" / "patch_000000_seed_30.json",
            )
            self.assertEqual(
                records[0]["audio_path"],
                root / "audio" / "patch_000000_seed_30.wav",
            )

            for record in records:
                self.assertTrue(record["patch_path"].exists())
                self.assertTrue(record["audio_path"].exists())
                self.assertIn("osc1_wave", load_patch(record["patch_path"]))

                info = sf.info(record["audio_path"])
                self.assertEqual(info.samplerate, DEFAULT_SAMPLE_RATE)
                self.assertEqual(info.channels, 1)

    def test_write_random_dataset_files_rejects_invalid_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_random_dataset_files(tmpdir, tmpdir, count=0)


if __name__ == "__main__":
    unittest.main()
