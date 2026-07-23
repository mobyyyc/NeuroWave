"""Train the current NeuroWave PyTorch inverse model on mel tensors."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.ml import save_metrics_report
from minisynth.reporting import compact_model_metrics
from minisynth.torch_model import (
    DEFAULT_LEARNING_RATE,
    LOSS_PRESET_NOISE_DETUNE,
    HEAD_MODE_GROUPED,
    MODEL_SIZE_LARGE,
    OPTIMIZER_ADAMW,
    POOLING_TIME_FREQUENCY,
    SCHEDULER_STEP,
    CHECKPOINT_BEST_VALIDATION,
    TARGET_MODE_MAIN_DETUNED_MIX,
    WAVEFORM_MODE_CLASSIFICATION,
    save_torch_checkpoint,
    train_inverse_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="v3.5_noise_detune_loss",
        help="Model identifier to store in the training metrics.",
    )
    parser.add_argument(
        "--tensor-data",
        required=True,
        help="Path to exported mel tensor NPZ data.",
    )
    parser.add_argument(
        "--validation-tensor-data",
        help="Optional separate sharded tensor directory for checkpoint selection.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epoch count.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=10,
        help="Epoch interval for the fixed step scheduler.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.5,
        help="Learning-rate multiplier for the step scheduler.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop after this many epochs without validation improvement.",
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
        default="models/v3.5_noise_detune_loss.pt",
        help="Path where the trained PyTorch checkpoint should be saved.",
    )
    parser.add_argument(
        "--metrics-output",
        default="runs/training/v3.5_noise_detune_loss_metrics.json",
        help="Path where training metrics JSON should be saved.",
    )
    return parser.parse_args()


def reject_benchmark_tensor_path(path, argument_name):
    if path is None:
        return
    if "benchmark" in {part.lower() for part in Path(path).parts}:
        raise ValueError(f"{argument_name} must not use an immutable benchmark partition")


def main() -> int:
    args = parse_args()
    reject_benchmark_tensor_path(args.tensor_data, "--tensor-data")
    reject_benchmark_tensor_path(args.validation_tensor_data, "--validation-tensor-data")
    result = train_inverse_model(
        tensor_path=args.tensor_data,
        validation_tensor_path=args.validation_tensor_data,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer_name=OPTIMIZER_ADAMW,
        weight_decay=args.weight_decay,
        scheduler_name=SCHEDULER_STEP,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_selection=CHECKPOINT_BEST_VALIDATION,
        model_size=MODEL_SIZE_LARGE,
        pooling_mode=POOLING_TIME_FREQUENCY,
        head_mode=HEAD_MODE_GROUPED,
        waveform_mode=WAVEFORM_MODE_CLASSIFICATION,
        target_mode=TARGET_MODE_MAIN_DETUNED_MIX,
        loss_preset=LOSS_PRESET_NOISE_DETUNE,
        random_state=args.random_state,
        device=args.device,
        progress=not args.quiet,
    )
    compact_metrics = compact_model_metrics(result["metrics"])
    model_path = save_torch_checkpoint(
        result["model"],
        args.model_output,
        metrics=compact_metrics,
    )
    output = {
        **compact_metrics,
        "model_path": str(model_path),
    }
    metrics_path = save_metrics_report(output, args.metrics_output)
    output["metrics_path"] = str(metrics_path)

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
