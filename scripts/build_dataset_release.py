"""Generate or export one declared partition of a NeuroWave dataset release."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import save_mel_tensor_dataset, write_random_dataset_files
from minisynth.dataset_release import load_dataset_release, partition_paths, partition_spec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "export"))
    parser.add_argument("--release", default="datasets/nwsd_v1.json")
    parser.add_argument("--partition", required=True, choices=("train", "dev", "benchmark"))
    parser.add_argument("--workers", type=int, help="Override the manifest worker count.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser.parse_args()


def ensure_empty_for_generation(paths):
    root = paths["root"]
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to mix generated data into existing partition: {root}. "
            "Use a new empty dataset release directory."
        )


def main():
    args = parse_args()
    release = load_dataset_release(args.release)
    paths = partition_paths(release, args.partition)
    spec = partition_spec(release, args.partition)
    generation = release["generation"]
    export = release["tensor_export"]

    if args.action == "generate":
        ensure_empty_for_generation(paths)
        workers = args.workers if args.workers is not None else generation["workers"]
        records = write_random_dataset_files(
            param_dir=paths["param_dir"],
            audio_dir=paths["audio_dir"],
            metadata_path=paths["metadata_path"],
            seed=spec["seed_start"],
            count=spec["count"],
            workers=workers,
            progress=not args.quiet and bool(generation.get("progress", False)),
            chunk_size=generation["chunk_size"],
        )
        result = {
            "action": "generate",
            "release": release["id"],
            "partition": args.partition,
            "count": len(records),
            "seed_start": spec["seed_start"],
            "seed_end": spec["seed_start"] + spec["count"] - 1,
            "metadata_path": str(paths["metadata_path"]),
        }
    else:
        if not paths["metadata_path"].exists():
            raise FileNotFoundError(f"Generate the partition before exporting tensors: {paths['metadata_path']}")
        workers = args.workers if args.workers is not None else export["workers"]
        saved = save_mel_tensor_dataset(
            metadata_path=paths["metadata_path"],
            output_path=paths["features_dir"],
            frames=export["frames"],
            workers=workers,
            progress=not args.quiet and bool(export.get("progress", False)),
            chunk_size=export["chunk_size"],
            shard_size=export["shard_size"],
        )
        saved_paths = saved if isinstance(saved, list) else [saved]
        result = {
            "action": "export",
            "release": release["id"],
            "partition": args.partition,
            "shard_count": len(saved_paths),
            "features_dir": str(paths["features_dir"]),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
