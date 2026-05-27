"""Render the current dark_saw patch as a basic smoke test."""

from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patches import PATCHES
from synth import SR, render_patch


def main() -> int:
    output_path = ROOT / "dark_saw.wav"
    audio = render_patch(**PATCHES["dark_saw"])

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    if len(audio) == 0:
        raise ValueError("Rendered audio is empty")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Rendered audio contains non-finite values")

    sf.write(output_path, audio, SR)
    info = sf.info(output_path)

    if info.samplerate != SR:
        raise ValueError(f"Expected sample rate {SR}, got {info.samplerate}")

    if info.frames != len(audio):
        raise ValueError(f"Expected {len(audio)} frames, got {info.frames}")

    print(f"Smoke render passed: {output_path}")
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
