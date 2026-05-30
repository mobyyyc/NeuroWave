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
        help="Path to write compressed NPZ tensors.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_MEL_TENSOR_FRAMES,
        help="Fixed mel-spectrogram time frames after crop/pad.",
    )
    return parser.parse_args()


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

    saved_path = save_mel_tensor_dataset(
        metadata_path=metadata_path,
        output_path=output_path,
        frames=args.frames,
    )
    with np.load(saved_path) as tensors:
        result = {
            "metadata_path": str(metadata_path),
            "output": str(saved_path),
            "features_shape": list(tensors["features"].shape),
            "targets_shape": list(tensors["targets"].shape),
            "frames": int(tensors["frames"]),
        }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
