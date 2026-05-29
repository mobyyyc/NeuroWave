"""Compare two audio files using current NeuroWave feature distances."""

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.compare import compare_audio_arrays


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Path to the target audio file.")
    parser.add_argument("candidate", help="Path to the candidate audio file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_audio, target_sample_rate = sf.read(args.target)
    candidate_audio, candidate_sample_rate = sf.read(args.candidate)

    result = compare_audio_arrays(
        target_audio,
        target_sample_rate,
        candidate_audio,
        candidate_sample_rate,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
