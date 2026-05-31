"""Evaluate a PyTorch model across multiple generated dataset clips."""

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.dataset import DEFAULT_METADATA_PATH, load_metadata, resolve_metadata_path
from minisynth.evaluation import evaluate_patch_prediction, summarize_weighted_distances
from minisynth.io import load_patch
from minisynth.torch_model import (
    DEFAULT_TORCH_MODEL_PATH,
    load_torch_checkpoint,
    predict_patch_from_audio,
)


DEFAULT_OUTPUT = Path("runs/evaluation/v3_pytorch_cnn_500seeds_on_d2_eval.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help="Path to generated dataset metadata JSONL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TORCH_MODEL_PATH,
        help="Path to a saved PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of clips to evaluate from the metadata file.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Metadata row index to start evaluating from.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write the evaluation JSON report.",
    )
    parser.add_argument(
        "--device",
        help="Optional torch device override, such as cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=0,
        help="Optional local search iterations after each PyTorch prediction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    rows = load_metadata(args.metadata)
    selected_rows = rows[args.start_index : args.start_index + args.count]
    if len(selected_rows) != args.count:
        raise ValueError("metadata does not contain enough rows for requested range")

    checkpoint = load_torch_checkpoint(args.model, device=args.device)
    clip_results = []
    total_clips = len(selected_rows)

    for offset, row in enumerate(selected_rows, start=1):
        print(f"\rEvaluating clip {offset}/{total_clips}", end="", flush=True)
        audio_path = resolve_metadata_path(args.metadata, row["audio_path"])
        target_audio, target_sample_rate = sf.read(audio_path)
        clip_result = {
            "index": row["index"],
            "seed": row["seed"],
            "audio_path": str(audio_path),
        }
        try:
            patch_path = resolve_metadata_path(args.metadata, row["patch_path"])
            target_patch = load_patch(patch_path)
            patch = predict_patch_from_audio(
                checkpoint["model"],
                target_audio,
                target_sample_rate,
                device=args.device,
                freq=target_patch.get("freq"),
            )
            result = evaluate_patch_prediction(
                patch,
                target_audio,
                target_sample_rate,
                refine_iterations=args.refine_iterations,
                refine_seed=row["seed"],
            )
            clip_result["comparison"] = result["comparison"]
        except ValueError as error:
            clip_result["error"] = str(error)

        clip_results.append(clip_result)
    print()

    successful_results = [result for result in clip_results if "comparison" in result]
    if not successful_results:
        raise ValueError("no clips produced valid predictions")

    report = {
        "metadata_path": str(args.metadata),
        "model_path": str(args.model),
        "model_metrics": checkpoint["metrics"],
        "start_index": args.start_index,
        "count": args.count,
        "refine_iterations": args.refine_iterations,
        "summary": {
            **summarize_weighted_distances(successful_results),
            "failed_count": len(clip_results) - len(successful_results),
        },
        "clips": clip_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(json.dumps({**report, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
