import unittest

from minisynth.dataset_release import partition_paths, validate_dataset_release


def sample_release():
    return {
        "id": "nwsd_v1",
        "root": "data/generated/nwsd_v1",
        "partitions": {
            "train": {"seed_start": 100000, "count": 5},
            "dev": {"seed_start": 200000, "count": 2},
            "benchmark": {"seed_start": 300000, "count": 3},
        },
    }


class TestDatasetRelease(unittest.TestCase):
    def test_valid_release_has_non_overlapping_partitions(self):
        self.assertTrue(validate_dataset_release(sample_release()))

    def test_release_rejects_overlapping_seed_ranges(self):
        release = sample_release()
        release["partitions"]["dev"]["seed_start"] = 100004

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_dataset_release(release)

    def test_partition_paths_are_nested_under_release_root(self):
        paths = partition_paths(sample_release(), "benchmark")

        self.assertEqual(paths["metadata_path"].as_posix(), "data/generated/nwsd_v1/benchmark/metadata.jsonl")
        self.assertEqual(paths["features_dir"].as_posix(), "data/generated/nwsd_v1/benchmark/features")
