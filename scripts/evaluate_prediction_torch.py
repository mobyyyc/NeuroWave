"""Predict with PyTorch, render the patch, and compare it against target audio."""

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.evaluation import evaluate_patch_prediction
from minisynth.io import save_patch
from minisynth.reporting import compact_model_metrics
from minisynth.torch_model import (
    load_torch_checkpoint,
    predict_patch_from_audio,
)


DEFAULT_OUTPUT_DIR = Path("runs/pytorch_prediction")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Path to the target audio clip.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a saved PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for predicted patch, rendered WAV, and report JSON.",
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
    output_dir = Path(args.output_dir)
    patch_path = output_dir / "predicted_patch.json"
    audio_path = output_dir / "predicted.wav"
    report_path = output_dir / "report.json"

    checkpoint = load_torch_checkpoint(args.model, device=args.device)
    target_audio, target_sample_rate = sf.read(args.target)
    patch = predict_patch_from_audio(
        checkpoint["model"],
        target_audio,
        target_sample_rate,
        device=args.device,
        freq=args.freq,
    )
    result = evaluate_patch_prediction(
        patch,
        target_audio,
        target_sample_rate,
    )

    save_patch(result["patch"], patch_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, result["rendered_audio"], DEFAULT_SAMPLE_RATE)

    report = {
        "target_audio": str(args.target),
        "model": str(args.model),
        "model_metrics": compact_model_metrics(checkpoint["metrics"]),
        "predicted_patch": str(patch_path),
        "predicted_audio": str(audio_path),
        "comparison": result["comparison"],
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(json.dumps({**report, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
