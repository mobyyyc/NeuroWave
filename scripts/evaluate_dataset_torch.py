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
from minisynth.evaluation import (
    evaluate_patch_prediction,
    parameter_error_report,
    patch_prediction_distribution,
    summarize_weighted_distances,
    worst_clip_diagnostics,
)
from minisynth.io import load_patch
from minisynth.oscillator_mix import (
    main_detuned_error_report,
    oscillator_mix_error_report,
    summarize_main_detuned_errors,
    summarize_oscillator_mix_errors,
)
from minisynth.reporting import (
    compact_clip_result,
    compact_model_metrics,
    compact_prediction_distribution,
)
from minisynth.torch_model import (
    load_torch_checkpoint,
    predict_patch_from_audio,
)


DEFAULT_OUTPUT = Path("runs/evaluation/eval.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help="Path to generated dataset metadata JSONL.",
    )
    parser.add_argument(
        "--model",
        required=True,
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
        "--diagnostics-top-n",
        type=int,
        default=10,
        help="Number of worst clips to summarize with target/predicted parameter errors.",
    )
    parser.add_argument(
        "--include-clips",
        action="store_true",
        help="Include every compact per-clip result in the report.",
    )
    parser.add_argument(
        "--include-full-clips",
        action="store_true",
        help="Include full per-clip patches, comparisons, and parameter errors.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-clip progress output.",
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
        if not args.quiet:
            print(f"\rEvaluating clip {offset}/{total_clips}", end="", flush=True)
        audio_path = resolve_metadata_path(args.metadata, row["audio_path"])
        target_audio, target_sample_rate = sf.read(audio_path)
        clip_result = {
            "index": row["index"],
            "seed": row["seed"],
            "audio_path": str(audio_path),
        }
        target_patch = None
        patch = None
        try:
            patch_path = resolve_metadata_path(args.metadata, row["patch_path"])
            target_patch = load_patch(patch_path)
            clip_result["target_patch"] = target_patch
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
            )
            clip_result["comparison"] = result["comparison"]
            clip_result["predicted_patch"] = result["patch"]
            clip_result["parameter_errors"] = parameter_error_report(
                target_patch,
                result["patch"],
            )
            clip_result["oscillator_mix_errors"] = oscillator_mix_error_report(
                target_patch,
                result["patch"],
            )
            clip_result["main_detuned_errors"] = main_detuned_error_report(
                target_patch,
                result["patch"],
            )
        except ValueError as error:
            clip_result["error"] = str(error)
            if patch is not None:
                clip_result["predicted_patch"] = patch
                if target_patch is not None:
                    clip_result["parameter_errors"] = parameter_error_report(
                        target_patch,
                        patch,
                    )
                    clip_result["oscillator_mix_errors"] = oscillator_mix_error_report(
                        target_patch,
                        patch,
                    )
                    clip_result["main_detuned_errors"] = main_detuned_error_report(
                        target_patch,
                        patch,
                    )

        clip_results.append(clip_result)
    if not args.quiet:
        print()

    successful_results = [result for result in clip_results if "comparison" in result]
    if not successful_results:
        raise ValueError("no clips produced valid predictions")

    report = {
        "metadata_path": str(args.metadata),
        "model_path": str(args.model),
        "model_metrics": compact_model_metrics(checkpoint["metrics"]),
        "start_index": args.start_index,
        "count": args.count,
        "summary": {
            **summarize_weighted_distances(successful_results),
            "failed_count": len(clip_results) - len(successful_results),
        },
        "diagnostics": {
            "prediction_distribution": compact_prediction_distribution(
                patch_prediction_distribution(successful_results)
            ),
            "oscillator_mix": summarize_oscillator_mix_errors(successful_results),
            "main_detuned": summarize_main_detuned_errors(successful_results),
            "worst_clips": worst_clip_diagnostics(
                successful_results,
                top_n=args.diagnostics_top_n,
                include_full=args.include_full_clips,
            ),
        },
    }
    if args.include_full_clips:
        report["clips"] = clip_results
    elif args.include_clips:
        report["clips"] = [compact_clip_result(result) for result in clip_results]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(json.dumps({**report, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
