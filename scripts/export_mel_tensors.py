"""Export generated dataset audio as fixed-size mel-spectrogram tensors."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import (
    DEFAULT_MEL_TENSOR_FRAMES,
    DEFAULT_METADATA_PATH,
    generated_dataset_paths,
    save_mel_tensor_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help="Path to generated dataset metadata JSONL.",
    )
    parser.add_argument(
        "--dataset-version",
        help="Generated dataset version under data/generated/. Used only when --metadata is omitted.",
    )
    parser.add_argument(
        "--output",
        help="Path to write compressed NPZ tensors. In sharded mode, this is used as the shard base path.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_MEL_TENSOR_FRAMES,
        help="Fixed mel-spectrogram time frames after crop/pad.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes for tensor export. Use 0 for conservative auto, 1 for serial.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to submit per multiprocessing chunk.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Rows per output shard. Use 0 for the original single-NPZ behavior.",
    )
    parser.add_argument(
        "--include-shards",
        action="store_true",
        help="Print every shard summary instead of only first/last shard paths.",
    )
    return parser.parse_args()


def tensor_file_summary(path):
    with np.load(path) as tensors:
        return {
            "output": str(path),
            "features_shape": list(tensors["features"].shape),
            "targets_shape": list(tensors["targets"].shape),
            "frames": int(tensors["frames"]),
        }


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata
    output_path = args.output

    if args.dataset_version and args.metadata == DEFAULT_METADATA_PATH:
        paths = generated_dataset_paths(args.dataset_version)
        metadata_path = paths["metadata_path"]
        output_path = output_path or paths["root"] / "features" / "mel_tensors.npz"

    if output_path is None:
        output_path = Path(metadata_path).parent / "features" / "mel_tensors.npz"

    saved = save_mel_tensor_dataset(
        metadata_path=metadata_path,
        output_path=output_path,
        frames=args.frames,
        workers=args.workers,
        progress=True,
        chunk_size=args.chunk_size,
        shard_size=args.shard_size,
    )

    if isinstance(saved, list):
        shard_summaries = [tensor_file_summary(path) for path in saved]
        result = {
            "metadata_path": str(metadata_path),
            "sharded": True,
            "shard_count": len(saved),
            "total_samples": sum(item["features_shape"][0] for item in shard_summaries),
            "frames": args.frames,
            "first_shard": shard_summaries[0] if shard_summaries else None,
            "last_shard": shard_summaries[-1] if shard_summaries else None,
        }
        if args.include_shards:
            result["shards"] = shard_summaries
    else:
        summary = tensor_file_summary(saved)
        result = {
            "metadata_path": str(metadata_path),
            "sharded": False,
            **summary,
        }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
