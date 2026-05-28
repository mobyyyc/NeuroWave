"""Generate one seeded random NeuroWave patch as JSON."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.io import save_patch
from minisynth.randomize import random_patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--output", help="Optional path to write the patch JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch = random_patch(args.seed)

    if args.output:
        save_patch(patch, args.output)
        print(f"Wrote random patch seed {args.seed} -> {args.output}")
    else:
        print(json.dumps(patch, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
