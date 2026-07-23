"""Generate seeded random NeuroWave patch JSON files."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import (
    DEFAULT_DATASET_VERSION,
    generated_dataset_paths,
    write_random_dataset_files,
)
from minisynth.io import load_patch, save_patch
from minisynth.randomize import random_patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--count", type=int, default=1, help="Number of patches to write.")
    parser.add_argument(
        "--dataset-version",
        default=DEFAULT_DATASET_VERSION,
        help="Generated dataset version directory under data/generated/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated patch JSON files.",
    )
    parser.add_argument(
        "--audio-output-dir",
        default=None,
        help="Directory for generated WAV files.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Path for generated metadata JSONL.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes for dataset generation. Use 0 for conservative auto, 1 for serial.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of samples to submit per multiprocessing chunk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-clip progress output; keep only the final summary.",
    )
    parser.add_argument("--output", help="Optional single patch JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output:
        if args.count != 1:
            raise ValueError("--output only supports --count 1")

        patch = random_patch(args.seed)
        save_patch(patch, args.output)
        print(f"Wrote random patch seed {args.seed} -> {args.output}")
    else:
        dataset_paths = generated_dataset_paths(args.dataset_version)
        output_dir = args.output_dir or dataset_paths["param_dir"]
        audio_output_dir = args.audio_output_dir or dataset_paths["audio_dir"]
        metadata_output = args.metadata_output or dataset_paths["metadata_path"]
        records = write_random_dataset_files(
            param_dir=output_dir,
            audio_dir=audio_output_dir,
            metadata_path=metadata_output,
            seed=args.seed,
            count=args.count,
            workers=args.workers,
            progress=args.count > 1 and not args.quiet,
            chunk_size=args.chunk_size,
        )

        if args.count == 1:
            print(json.dumps(load_patch(records[0]["patch_path"]), indent=2))
        else:
            print(
                f"Wrote {len(records)} random patches -> {output_dir} "
                f"WAVs -> {audio_output_dir} "
                f"and metadata -> {metadata_output}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
