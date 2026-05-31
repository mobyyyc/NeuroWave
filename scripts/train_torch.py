"""Train the first NeuroWave PyTorch CNN inverse model on mel tensors."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.ml import save_metrics_report
from minisynth.torch_model import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BENCHMARK_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TORCH_METRICS_PATH,
    DEFAULT_TORCH_MODEL_ID,
    DEFAULT_TORCH_MODEL_PATH,
    DEFAULT_TORCH_TENSOR_PATH,
    DEFAULT_TARGET_MODE,
    DEFAULT_WAVEFORM_MODE,
    TARGET_MODE_FULL,
    TARGET_MODE_PITCH_CONDITIONED_TIMBRE,
    WAVEFORM_MODE_CLASSIFICATION,
    WAVEFORM_MODE_SCALAR,
    save_torch_checkpoint,
    train_inverse_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_TORCH_MODEL_ID,
        help="Model identifier to store in the training metrics.",
    )
    parser.add_argument(
        "--tensor-data",
        default=DEFAULT_TORCH_TENSOR_PATH,
        help="Path to exported mel tensor NPZ data.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epoch count.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of examples reserved for validation metrics.",
    )
    parser.add_argument(
        "--benchmark-size",
        type=float,
        default=DEFAULT_BENCHMARK_SIZE,
        help="Fraction of examples reserved for fixed benchmark metrics.",
    )
    parser.add_argument(
        "--waveform-mode",
        choices=(WAVEFORM_MODE_CLASSIFICATION, WAVEFORM_MODE_SCALAR),
        default=DEFAULT_WAVEFORM_MODE,
        help="How waveform parameters are trained.",
    )
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_FULL, TARGET_MODE_PITCH_CONDITIONED_TIMBRE),
        default=DEFAULT_TARGET_MODE,
        help="Which synth parameters are used as model outputs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for train/test split and model initialization.",
    )
    parser.add_argument(
        "--device",
        help="Optional torch device override, such as cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable epoch/batch progress output.",
    )
    parser.add_argument(
        "--model-output",
        default=DEFAULT_TORCH_MODEL_PATH,
        help="Path where the trained PyTorch checkpoint should be saved.",
    )
    parser.add_argument(
        "--metrics-output",
        default=DEFAULT_TORCH_METRICS_PATH,
        help="Path where training metrics JSON should be saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_inverse_model(
        tensor_path=args.tensor_data,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        benchmark_size=args.benchmark_size,
        waveform_mode=args.waveform_mode,
        target_mode=args.target_mode,
        random_state=args.random_state,
        device=args.device,
        progress=not args.quiet,
    )
    model_path = save_torch_checkpoint(
        result["model"],
        args.model_output,
        metrics=result["metrics"],
    )
    output = {
        **result["metrics"],
        "model_path": str(model_path),
    }
    metrics_path = save_metrics_report(output, args.metrics_output)
    output["metrics_path"] = str(metrics_path)

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
