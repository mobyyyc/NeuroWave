"""Generate seeded random NeuroWave patch JSON files."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import DEFAULT_PARAM_DIR, write_random_patch_files
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
        paths = write_random_patch_files(
            output_dir=args.output_dir,
            seed=args.seed,
            count=args.count,
        )

        if args.count == 1:
            print(json.dumps(load_patch(paths[0]), indent=2))
        else:
            print(f"Wrote {len(paths)} random patches -> {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
