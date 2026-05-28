"""Render the current dark_saw patch as a basic smoke test."""

from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.engine import render_patch
from minisynth.io import load_patch


def main() -> int:
    output_path = ROOT / "dark_saw.wav"
    patch = load_patch(ROOT / "presets" / "dark_saw.json")
    audio = render_patch(**patch)

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    if len(audio) == 0:
        raise ValueError("Rendered audio is empty")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Rendered audio contains non-finite values")

    sf.write(output_path, audio, DEFAULT_SAMPLE_RATE)
    info = sf.info(output_path)

    if info.samplerate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {DEFAULT_SAMPLE_RATE}, got {info.samplerate}"
        )

    if info.frames != len(audio):
        raise ValueError(f"Expected {len(audio)} frames, got {info.frames}")

    print(f"Smoke render passed: {output_path}")
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
