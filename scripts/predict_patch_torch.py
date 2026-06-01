"""Predict a NeuroWave patch JSON from one audio clip using a PyTorch model."""

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.io import save_patch
from minisynth.reporting import compact_model_metrics
from minisynth.torch_model import (
    load_torch_checkpoint,
    predict_patch_from_audio,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Path to the input audio clip.")
    parser.add_argument("output", help="Path to write the predicted patch JSON.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a saved PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--device",
        help="Optional torch device override, such as cpu or cuda.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        help="Known or estimated fundamental frequency in Hz for pitch-conditioned models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = load_torch_checkpoint(args.model, device=args.device)
    audio, sample_rate = sf.read(args.audio)
    patch = predict_patch_from_audio(
        checkpoint["model"],
        audio,
        sample_rate,
        device=args.device,
        freq=args.freq,
    )

    save_patch(patch, args.output)
    print(
        json.dumps(
            {
                "audio": str(args.audio),
                "model": str(args.model),
                "output": str(args.output),
                "model_metrics": compact_model_metrics(checkpoint["metrics"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
