"""Validate generated partition boundaries and tensor shards for a dataset release."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import load_metadata
from minisynth.dataset_release import load_dataset_release, partition_paths, partition_spec
from minisynth.torch_model import load_mel_tensor_shard_source


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="datasets/nwsd_v1.json")
    parser.add_argument("--partition", required=True, choices=("train", "dev", "benchmark"))
    parser.add_argument("--require-tensors", action="store_true")
    return parser.parse_args()


def validate_metadata(paths, spec):
    metadata_path = paths["metadata_path"]
    rows = load_metadata(metadata_path)
    if len(rows) != spec["count"]:
        raise ValueError(f"Expected {spec['count']} metadata rows, got {len(rows)}")

    expected_seeds = np.arange(spec["seed_start"], spec["seed_start"] + spec["count"])
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    indices = np.asarray([row["index"] for row in rows], dtype=np.int64)
    if not np.array_equal(seeds, expected_seeds):
        raise ValueError("metadata seed range is not contiguous or does not match the manifest")
    if not np.array_equal(indices, np.arange(spec["count"])):
        raise ValueError("metadata indices are not contiguous from zero")
    for row in rows:
        for key in ("patch_path", "audio_path"):
            candidate = Path(row[key])
            if not candidate.exists():
                candidate = metadata_path.parent / candidate
            if not candidate.exists():
                raise FileNotFoundError(f"metadata references missing {key}: {row[key]}")
    return len(rows)


def main():
    args = parse_args()
    release = load_dataset_release(args.release)
    paths = partition_paths(release, args.partition)
    spec = partition_spec(release, args.partition)
    sample_count = validate_metadata(paths, spec)
    result = {
        "release": release["id"],
        "partition": args.partition,
        "metadata_samples": sample_count,
        "seed_start": spec["seed_start"],
        "seed_end": spec["seed_start"] + spec["count"] - 1,
    }
    if args.require_tensors:
        source = load_mel_tensor_shard_source(paths["features_dir"])
        if source["sample_count"] != sample_count:
            raise ValueError("tensor shard sample count does not match metadata")
        result.update(
            {
                "tensor_shards": len(source["shard_paths"]),
                "tensor_samples": source["sample_count"],
                "feature_shape": list(source["feature_shape"]),
                "target_dim": source["target_dim"],
            }
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
