"""Generate a random patch, predict it from audio, and render both versions."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.io import save_patch
from minisynth.randomize import random_patch
from minisynth.reporting import compact_model_metrics
from minisynth.torch_model import (
    DEFAULT_MEL_TENSOR_FRAMES,
    DEFAULT_TORCH_MODEL_PATH,
    load_torch_checkpoint,
    predict_patch_from_audio,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_TORCH_MODEL_PATH,
        help="Path to a saved PyTorch checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random patch seed. Defaults to a timestamp-derived seed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the two patch JSON files and two WAV files.",
    )
    parser.add_argument(
        "--device",
        help="Optional torch device override, such as cpu or cuda.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_MEL_TENSOR_FRAMES,
        help="Mel frame count used by the model input.",
    )
    return parser.parse_args()


def timestamp_seed():
    return int(datetime.now().strftime("%Y%m%d%H%M%S"))


def render_audio(patch):
    audio = render_patch(**patch)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    if len(audio) == 0:
        raise ValueError("Rendered audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Rendered audio contains non-finite values")
    return audio


def write_wav(path, audio):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, DEFAULT_SAMPLE_RATE)
    return destination


def default_output_dir(seed):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("playground") / f"predict_render_{stamp}_seed_{seed}"


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else timestamp_seed()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_patch = random_patch(seed)
    target_audio = render_audio(target_patch)

    target_patch_path = output_dir / "target_patch.json"
    target_audio_path = output_dir / "target.wav"
    predicted_patch_path = output_dir / "predicted_patch.json"
    predicted_audio_path = output_dir / "predicted.wav"
    summary_path = output_dir / "summary.json"

    save_patch(target_patch, target_patch_path)
    write_wav(target_audio_path, target_audio)

    checkpoint = load_torch_checkpoint(args.model, device=args.device)
    predicted_patch = predict_patch_from_audio(
        checkpoint["model"],
        target_audio,
        DEFAULT_SAMPLE_RATE,
        device=args.device,
        frames=args.frames,
        freq=target_patch["freq"],
    )
    predicted_audio = render_audio(predicted_patch)

    save_patch(predicted_patch, predicted_patch_path)
    write_wav(predicted_audio_path, predicted_audio)

    summary = {
        "seed": seed,
        "model": str(args.model),
        "output_dir": str(output_dir),
        "target_patch": str(target_patch_path),
        "target_wav": str(target_audio_path),
        "predicted_patch": str(predicted_patch_path),
        "predicted_wav": str(predicted_audio_path),
        "freq_context_hz": float(target_patch["freq"]),
        "model_metrics": compact_model_metrics(checkpoint["metrics"]),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
