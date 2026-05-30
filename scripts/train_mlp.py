"""Train the first NeuroWave MLP baseline on generated dataset metadata."""

import argparse
import json
from pathlib import Path
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import DEFAULT_METADATA_PATH
from minisynth.ml import (
    DEFAULT_MODEL_PATH,
    save_metrics_report,
    save_model_checkpoint,
    train_mlp_from_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help="Path to generated metadata JSONL.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Hidden layer size for the single-layer MLP baseline.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum MLP training iterations.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of examples reserved for test metrics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for train/test split and MLP initialization.",
    )
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_PATH,
        help="Path where the trained checkpoint should be saved.",
    )
    parser.add_argument(
        "--metrics-output",
        help="Optional path where training metrics JSON should be saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = train_mlp_from_metadata(
            args.metadata,
            hidden_layer_sizes=(args.hidden_size,),
            max_iter=args.max_iter,
            random_state=args.random_state,
            test_size=args.test_size,
        )

    model_path = save_model_checkpoint(
        result["model"],
        args.model_output,
        metrics=result["metrics"],
    )
    output = {
        **result["metrics"],
        "model_path": str(model_path),
    }
    if args.metrics_output:
        metrics_path = save_metrics_report(output, args.metrics_output)
        output["metrics_path"] = str(metrics_path)

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
