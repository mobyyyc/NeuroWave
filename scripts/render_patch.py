"""Render a NeuroWave JSON preset to a WAV file."""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", help="Path to a JSON preset file.")
    parser.add_argument("output", help="Path to write the rendered WAV file.")
    return parser.parse_args()


def render_preset(preset, output):
    patch = load_patch(preset)
    audio = render_patch(**patch)

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    if len(audio) == 0:
        raise ValueError("Rendered audio is empty")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Rendered audio contains non-finite values")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, DEFAULT_SAMPLE_RATE)

    print(f"Rendered {preset} -> {output_path}")


def main() -> int:
    args = parse_args()
    render_preset(args.preset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
