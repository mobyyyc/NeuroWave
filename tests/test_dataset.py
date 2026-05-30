import json
import tempfile
import unittest
from pathlib import Path

import soundfile as sf
import numpy as np

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.dataset import (
    DEFAULT_METADATA_PATH,
    audio_filename,
    audio_feature_vector,
    generated_dataset_paths,
    load_metadata,
    load_training_dataset,
    metadata_record,
    patch_filename,
    resolve_metadata_path,
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
                metadata_path=root / "metadata.jsonl",
                seed=30,
                count=2,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["index"], 0)
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
                self.assertEqual(record["sample_rate"], DEFAULT_SAMPLE_RATE)

                info = sf.info(record["audio_path"])
                self.assertEqual(info.samplerate, DEFAULT_SAMPLE_RATE)
                self.assertEqual(info.channels, 1)
                self.assertEqual(record["frames"], info.frames)

    def test_write_random_dataset_files_saves_metadata_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.jsonl"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=50,
                count=2,
            )

            rows = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["index"], 0)
            self.assertEqual(rows[0]["seed"], 50)
            self.assertEqual(
                rows[0]["patch_path"],
                str(root / "params" / "patch_000000_seed_50.json"),
            )
            self.assertEqual(
                rows[0]["audio_path"],
                str(root / "audio" / "patch_000000_seed_50.wav"),
            )
            self.assertEqual(rows[0]["sample_rate"], DEFAULT_SAMPLE_RATE)
            self.assertGreater(rows[0]["frames"], 0)

    def test_write_random_dataset_files_is_reproducible_for_same_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_root = root / "first"
            second_root = root / "second"

            first_records = write_random_dataset_files(
                param_dir=first_root / "params",
                audio_dir=first_root / "audio",
                metadata_path=first_root / "metadata.jsonl",
                seed=70,
                count=2,
            )
            second_records = write_random_dataset_files(
                param_dir=second_root / "params",
                audio_dir=second_root / "audio",
                metadata_path=second_root / "metadata.jsonl",
                seed=70,
                count=2,
            )

            for first_record, second_record in zip(first_records, second_records):
                self.assertEqual(first_record["seed"], second_record["seed"])
                self.assertEqual(
                    load_patch(first_record["patch_path"]),
                    load_patch(second_record["patch_path"]),
                )

            self.assertEqual(
                first_records[0]["frames"],
                second_records[0]["frames"],
            )

    def test_metadata_record_converts_paths_to_strings(self):
        record = {
            "index": 0,
            "seed": 1,
            "patch_path": Path("params/example.json"),
            "audio_path": Path("audio/example.wav"),
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "frames": 100,
        }

        self.assertEqual(
            metadata_record(record),
            {
                "index": 0,
                "seed": 1,
                "patch_path": "params/example.json",
                "audio_path": "audio/example.wav",
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "frames": 100,
            },
        )

    def test_default_metadata_path_targets_generated_dataset_root(self):
        self.assertEqual(DEFAULT_METADATA_PATH, Path("data/generated/v1/metadata.jsonl"))

    def test_generated_dataset_paths_targets_versioned_dataset_root(self):
        paths = generated_dataset_paths("v2")

        self.assertEqual(paths["root"], Path("data/generated/v2"))
        self.assertEqual(paths["param_dir"], Path("data/generated/v2/params"))
        self.assertEqual(paths["audio_dir"], Path("data/generated/v2/audio"))
        self.assertEqual(
            paths["metadata_path"],
            Path("data/generated/v2/metadata.jsonl"),
        )

    def test_write_random_dataset_files_rejects_invalid_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_random_dataset_files(tmpdir, tmpdir, count=0)

    def test_load_metadata_reads_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.jsonl"
            metadata_path.write_text(
                '{"index": 0, "seed": 1}\n\n{"index": 1, "seed": 2}\n',
                encoding="utf-8",
            )

            rows = load_metadata(metadata_path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["seed"], 1)
            self.assertEqual(rows[1]["seed"], 2)

    def test_resolve_metadata_path_uses_metadata_parent_for_relative_paths(self):
        resolved = resolve_metadata_path(
            Path("data/generated/v1/metadata.jsonl"),
            "audio/does-not-exist.wav",
        )

        self.assertEqual(resolved, Path("data/generated/v1/audio/does-not-exist.wav"))

    def test_resolve_metadata_path_keeps_existing_repo_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "example.wav"
            existing.touch()
            current_dir = Path.cwd()

            try:
                import os

                os.chdir(root)
                resolved = resolve_metadata_path(
                    Path("data/generated/v1/metadata.jsonl"),
                    "example.wav",
                )
            finally:
                os.chdir(current_dir)

            self.assertEqual(resolved, Path("example.wav"))

    def test_audio_feature_vector_returns_compact_numeric_features(self):
        audio = np.sin(
            2 * np.pi * 440.0 * np.arange(DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        )

        features = audio_feature_vector(audio, DEFAULT_SAMPLE_RATE)

        self.assertEqual(features.shape, (7,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_load_training_dataset_returns_features_and_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.jsonl"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=90,
                count=2,
            )

            features, targets = load_training_dataset(metadata_path)

            self.assertEqual(features.shape, (2, 7))
            self.assertEqual(targets.shape[0], 2)
            self.assertTrue(np.all(np.isfinite(features)))
            self.assertTrue(np.all(targets >= 0.0))
            self.assertTrue(np.all(targets <= 1.0))

    def test_load_training_dataset_rejects_empty_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.jsonl"
            metadata_path.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_training_dataset(metadata_path)


if __name__ == "__main__":
    unittest.main()
