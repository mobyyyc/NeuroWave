"""Predict a patch from an external WAV and render the predicted audio."""

import argparse
import json
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
from minisynth.reporting import compact_model_metrics
from minisynth.torch_model import (
    DEFAULT_MEL_TENSOR_FRAMES,
    load_torch_checkpoint,
    predict_patch_from_audio,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        help="Path to the input WAV file.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a saved PyTorch checkpoint.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        required=True,
        help="Known or estimated fundamental frequency in Hz.",
    )
    parser.add_argument(
        "--output-dir",
        default="playground",
        help="Directory for predicted JSON, predicted WAV, and summary.",
    )
    parser.add_argument(
        "--prefix",
        help="Output filename prefix. Defaults to the input audio filename stem.",
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


def load_mono_audio(path):
    audio, sample_rate = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}")
    if len(audio) == 0:
        raise ValueError("Input audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Input audio contains non-finite values")
    return audio, sample_rate


def render_audio(patch):
    audio = render_patch(**patch)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono rendered audio, got shape {audio.shape}")
    if len(audio) == 0:
        raise ValueError("Rendered audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Rendered audio contains non-finite values")
    return audio


def main() -> int:
    args = parse_args()
    input_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem

    predicted_patch_path = output_dir / f"{prefix}_predicted.json"
    predicted_audio_path = output_dir / f"{prefix}_predicted.wav"
    summary_path = output_dir / f"{prefix}_summary.json"

    source_audio, source_sample_rate = load_mono_audio(input_path)
    checkpoint = load_torch_checkpoint(args.model, device=args.device)
    predicted_patch = predict_patch_from_audio(
        checkpoint["model"],
        source_audio,
        source_sample_rate,
        device=args.device,
        frames=args.frames,
        freq=args.freq,
    )
    predicted_audio = render_audio(predicted_patch)

    save_patch(predicted_patch, predicted_patch_path)
    sf.write(predicted_audio_path, predicted_audio, DEFAULT_SAMPLE_RATE)

    summary = {
        "input_audio": str(input_path),
        "input_sample_rate": int(source_sample_rate),
        "model": str(args.model),
        "freq_context_hz": float(args.freq),
        "predicted_patch": str(predicted_patch_path),
        "predicted_wav": str(predicted_audio_path),
        "model_metrics": compact_model_metrics(checkpoint["metrics"]),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
