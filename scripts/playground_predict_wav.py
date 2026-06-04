"""Predict a patch from an external WAV and write an app-style run folder."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.app_inference import AppInferenceRequest, run_app_inference
from minisynth.torch_model import (
    DEFAULT_MEL_TENSOR_FRAMES,
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
        help="Directory for the app-style run folder.",
    )
    parser.add_argument(
        "--prefix",
        help="Run folder name. Defaults to a timestamp plus input audio filename stem.",
    )
    parser.add_argument(
        "--crop-start",
        type=float,
        default=0.0,
        help="Crop start in seconds.",
    )
    parser.add_argument(
        "--crop-end",
        type=float,
        help="Crop end in seconds. Defaults to the end of the file.",
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


def main() -> int:
    args = parse_args()
    result = run_app_inference(
        AppInferenceRequest(
            audio_path=str(Path(args.audio)),
            model_path=str(Path(args.model)),
            freq_hz=args.freq,
            crop_start_seconds=args.crop_start,
            crop_end_seconds=args.crop_end,
            output_dir=args.output_dir,
            run_id=args.prefix,
            device=args.device,
            frames=args.frames,
        )
    )

    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
