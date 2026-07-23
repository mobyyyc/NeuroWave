"""Helpers for reproducible, partitioned NeuroWave dataset releases."""

import json
from pathlib import Path


REQUIRED_PARTITIONS = ("train", "dev", "benchmark")


def load_dataset_release(path):
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_dataset_release(payload, source)
    return payload


def validate_dataset_release(payload, source=None):
    if not isinstance(payload, dict):
        raise ValueError("dataset release manifest must be a JSON object")
    if not payload.get("id"):
        raise ValueError("dataset release manifest requires an id")
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("dataset release manifest requires partitions")

    intervals = []
    for name in REQUIRED_PARTITIONS:
        partition = partitions.get(name)
        if not isinstance(partition, dict):
            raise ValueError(f"dataset release manifest requires {name} partition")
        seed_start = partition.get("seed_start")
        count = partition.get("count")
        if not isinstance(seed_start, int) or not isinstance(count, int) or count < 1:
            raise ValueError(f"partition {name} requires integer seed_start and positive count")
        intervals.append((seed_start, seed_start + count, name))

    for start, end, name in intervals:
        for other_start, other_end, other_name in intervals:
            if name != other_name and start < other_end and other_start < end:
                raise ValueError(f"seed ranges overlap: {name} and {other_name}")

    root = payload.get("root")
    if not isinstance(root, str) or not root:
        raise ValueError("dataset release manifest requires a root")
    return True


def partition_paths(payload, partition_name):
    partition = payload["partitions"].get(partition_name)
    if partition is None:
        raise ValueError(f"unknown dataset partition: {partition_name}")
    root = Path(payload["root"]) / partition_name
    return {
        "root": root,
        "param_dir": root / "params",
        "audio_dir": root / "audio",
        "metadata_path": root / "metadata.jsonl",
        "features_dir": root / "features",
    }


def partition_spec(payload, partition_name):
    partition = payload["partitions"].get(partition_name)
    if partition is None:
        raise ValueError(f"unknown dataset partition: {partition_name}")
    return partition
