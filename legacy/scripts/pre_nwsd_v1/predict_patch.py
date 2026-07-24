"""Predict a NeuroWave patch JSON from one audio clip."""

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.io import save_patch
from minisynth.ml import DEFAULT_MODEL_PATH, load_model_checkpoint, predict_patch_from_audio


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Path to the input audio clip.")
    parser.add_argument("output", help="Path to write the predicted patch JSON.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to a saved model checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = load_model_checkpoint(args.model)
    audio, sample_rate = sf.read(args.audio)
    patch = predict_patch_from_audio(checkpoint["model"], audio, sample_rate)

    save_patch(patch, args.output)
    print(
        json.dumps(
            {
                "audio": str(args.audio),
                "model": str(args.model),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
