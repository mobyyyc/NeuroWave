"""Evaluate a PyTorch checkpoint on the fixed NeuroWave product benchmark."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.dataset import audio_filename, patch_filename
from minisynth.evaluation import (
    evaluate_patch_prediction,
    parameter_error_report,
    summarize_weighted_distances,
    worst_clip_diagnostics,
)
from minisynth.io import load_patch, save_patch
from minisynth.oscillator_mix import main_detuned_error_report, oscillator_mix_error_report
from minisynth.product_benchmark import load_product_benchmark
from minisynth.reporting import compact_model_metrics
from minisynth.torch_model import load_torch_checkpoint, predict_patch_from_audio


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="datasets/product_benchmark_v1.json")
    parser.add_argument("--model", required=True, help="Path to a PyTorch checkpoint.")
    parser.add_argument("--output-dir", help="Empty directory for this evaluation run.")
    parser.add_argument("--device", help="Optional torch device override, such as cpu or cuda.")
    parser.add_argument("--diagnostics-top-n", type=int, default=10)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress output.")
    return parser.parse_args()


def repository_revision():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def default_output_dir(benchmark_id, model_path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs/nwsd_v1/evaluation/product_benchmark") / benchmark_id / Path(model_path).stem / stamp


def ensure_empty_output_dir(path):
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to mix product-benchmark outputs into non-empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def case_source_paths(benchmark, case):
    source = case["source"]
    root = Path(benchmark["source_dataset"]["root"]) / source["partition"]
    return {
        "target_audio": root / "audio" / audio_filename(source["index"], source["seed"]),
        "target_patch": root / "params" / patch_filename(source["index"], source["seed"]),
    }


def category_summaries(results, category_ids):
    grouped = defaultdict(list)
    for result in results:
        for category in result["categories"]:
            grouped[category].append(result)

    summaries = {}
    for category in category_ids:
        category_results = grouped[category]
        successful = [result for result in category_results if "comparison" in result]
        summary = {
            "case_count": len(category_results),
            "successful_count": len(successful),
            "failed_count": len(category_results) - len(successful),
            "worst_case_ids": [
                result["case_id"]
                for result in sorted(
                    successful,
                    key=lambda result: result["comparison"]["weighted_distance"],
                    reverse=True,
                )[:3]
            ],
        }
        if successful:
            summary.update(summarize_weighted_distances(successful))
        summaries[category] = summary
    return summaries


def ranked_failure_groups(category_metrics):
    groups = []
    for category, metrics in category_metrics.items():
        group = {"category": category, **metrics}
        groups.append(group)
    return sorted(
        groups,
        key=lambda group: (
            group.get("mean_weighted_distance", float("-inf")),
            group["failed_count"],
        ),
        reverse=True,
    )


def evaluate_case(benchmark, case, checkpoint, output_dir, device):
    sources = case_source_paths(benchmark, case)
    target_audio_path = sources["target_audio"]
    target_patch_path = sources["target_patch"]
    if not target_audio_path.exists() or not target_patch_path.exists():
        raise FileNotFoundError(f"Missing benchmark asset for {case['id']}: {target_audio_path}")

    case_dir = output_dir / "cases" / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    target_audio, target_sample_rate = sf.read(target_audio_path)
    target_patch = load_patch(target_patch_path)
    prediction = predict_patch_from_audio(
        checkpoint["model"],
        target_audio,
        target_sample_rate,
        device=device,
        freq=case["pitch_context_hz"],
    )
    evaluation = evaluate_patch_prediction(prediction, target_audio, target_sample_rate)

    target_copy_path = case_dir / "target.wav"
    prediction_audio_path = case_dir / "predicted.wav"
    target_patch_copy_path = case_dir / "target_patch.json"
    prediction_patch_path = case_dir / "predicted_patch.json"
    sf.write(target_copy_path, target_audio, target_sample_rate)
    sf.write(prediction_audio_path, evaluation["rendered_audio"], DEFAULT_SAMPLE_RATE)
    save_patch(target_patch, target_patch_copy_path)
    save_patch(evaluation["patch"], prediction_patch_path)

    return {
        "case_id": case["id"],
        "categories": case["categories"],
        "known_limitation": case["known_limitation"],
        "source": case["source"],
        "pitch_context_hz": case["pitch_context_hz"],
        "index": case["source"]["index"],
        "seed": case["source"]["seed"],
        "audio_path": str(target_audio_path),
        "artifacts": {
            "target_audio": str(target_copy_path.relative_to(output_dir)),
            "predicted_audio": str(prediction_audio_path.relative_to(output_dir)),
            "target_patch": str(target_patch_copy_path.relative_to(output_dir)),
            "predicted_patch": str(prediction_patch_path.relative_to(output_dir)),
        },
        "comparison": evaluation["comparison"],
        "target_patch": target_patch,
        "predicted_patch": evaluation["patch"],
        "parameter_errors": parameter_error_report(target_patch, evaluation["patch"]),
        "oscillator_mix_errors": oscillator_mix_error_report(target_patch, evaluation["patch"]),
        "main_detuned_errors": main_detuned_error_report(target_patch, evaluation["patch"]),
    }


def main():
    args = parse_args()
    if args.diagnostics_top_n < 1:
        raise ValueError("--diagnostics-top-n must be at least 1")
    benchmark = load_product_benchmark(args.benchmark)
    output_dir = ensure_empty_output_dir(
        args.output_dir or default_output_dir(benchmark["id"], args.model)
    )
    checkpoint = load_torch_checkpoint(args.model, device=args.device)
    case_results = []
    for number, case in enumerate(benchmark["cases"], start=1):
        if not args.quiet:
            print(f"Evaluating product benchmark case {number}/{len(benchmark['cases'])}: {case['id']}")
        try:
            case_results.append(evaluate_case(benchmark, case, checkpoint, output_dir, args.device))
        except (OSError, ValueError) as error:
            case_results.append(
                {
                    "case_id": case["id"],
                    "categories": case["categories"],
                    "source": case["source"],
                    "pitch_context_hz": case["pitch_context_hz"],
                    "index": case["source"]["index"],
                    "seed": case["source"]["seed"],
                    "error": str(error),
                }
            )

    successful = [result for result in case_results if "comparison" in result]
    if not successful:
        raise ValueError("no product benchmark cases produced valid predictions")
    category_ids = [category["id"] for category in benchmark["categories"]]
    categories = category_summaries(case_results, category_ids)
    report = {
        "benchmark_id": benchmark["id"],
        "benchmark_path": str(args.benchmark),
        "model_path": str(args.model),
        "repository_revision": repository_revision(),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": {"pitch_context": "manifest pitch_context_hz", "mel_frames": 256},
        "model_metrics": compact_model_metrics(checkpoint["metrics"]),
        "summary": {
            **summarize_weighted_distances(successful),
            "failed_count": len(case_results) - len(successful),
        },
        "category_metrics": categories,
        "ranked_failure_groups": ranked_failure_groups(categories),
        "worst_cases": worst_clip_diagnostics(
            successful,
            top_n=args.diagnostics_top_n,
            include_full=True,
        ),
        "cases": case_results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "benchmark_id": benchmark["id"],
                "summary": report["summary"],
                "report": str(report_path),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
