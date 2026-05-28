"""Generate seeded random NeuroWave patch JSON files."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_PARAM_DIR,
    write_random_dataset_files,
)
from minisynth.io import load_patch, save_patch
from minisynth.randomize import random_patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--count", type=int, default=1, help="Number of patches to write.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PARAM_DIR,
        help="Directory for generated patch JSON files.",
    )
    parser.add_argument(
        "--audio-output-dir",
        default=DEFAULT_AUDIO_DIR,
        help="Directory for generated WAV files.",
    )
    parser.add_argument(
        "--metadata-output",
        default=DEFAULT_METADATA_PATH,
        help="Path for generated metadata JSONL.",
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
        records = write_random_dataset_files(
            param_dir=args.output_dir,
            audio_dir=args.audio_output_dir,
            metadata_path=args.metadata_output,
            seed=args.seed,
            count=args.count,
        )

        if args.count == 1:
            print(json.dumps(load_patch(records[0]["patch_path"]), indent=2))
        else:
            print(
                f"Wrote {len(records)} random patches -> {args.output_dir} "
                f"WAVs -> {args.audio_output_dir} "
                f"and metadata -> {args.metadata_output}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
