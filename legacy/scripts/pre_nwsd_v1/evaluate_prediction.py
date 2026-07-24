"""Predict a patch, render it, and compare it against a target audio clip."""

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.evaluation import evaluate_audio_prediction
from minisynth.io import save_patch
from minisynth.ml import DEFAULT_MODEL_PATH, load_model_checkpoint


DEFAULT_OUTPUT_DIR = Path("runs/mlp_prediction")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Path to the target audio clip.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to a saved model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for predicted patch, rendered WAV, and report JSON.",
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=0,
        help="Optional local search iterations after the ML prediction.",
    )
    parser.add_argument(
        "--refine-seed",
        type=int,
        default=0,
        help="Random seed for optional local refinement.",
    )
    parser.add_argument(
        "--refine-step-size",
        type=float,
        default=0.05,
        help="Normalized parameter perturbation size for optional local refinement.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    patch_path = output_dir / "predicted_patch.json"
    audio_path = output_dir / "predicted.wav"
    report_path = output_dir / "report.json"

    checkpoint = load_model_checkpoint(args.model)
    target_audio, target_sample_rate = sf.read(args.target)
    result = evaluate_audio_prediction(
        checkpoint["model"],
        target_audio,
        target_sample_rate,
        refine_iterations=args.refine_iterations,
        refine_seed=args.refine_seed,
        refine_step_size=args.refine_step_size,
    )

    save_patch(result["patch"], patch_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, result["rendered_audio"], DEFAULT_SAMPLE_RATE)

    report = {
        "target_audio": str(args.target),
        "model": str(args.model),
        "predicted_patch": str(patch_path),
        "predicted_audio": str(audio_path),
        "comparison": result["comparison"],
    }
    if "refinement" in result:
        report["refinement"] = result["refinement"]

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(json.dumps({**report, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
