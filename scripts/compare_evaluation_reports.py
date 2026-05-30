"""Compare two NeuroWave dataset evaluation reports."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Path to baseline evaluation report JSON.")
    parser.add_argument("candidate", help="Path to candidate evaluation report JSON.")
    parser.add_argument(
        "--output",
        help="Optional path to write comparison JSON.",
    )
    return parser.parse_args()


def load_report(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def compare_reports(baseline, candidate):
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    baseline_mean = baseline_summary["mean_weighted_distance"]
    candidate_mean = candidate_summary["mean_weighted_distance"]
    baseline_median = baseline_summary["median_weighted_distance"]
    candidate_median = candidate_summary["median_weighted_distance"]

    return {
        "baseline_model": baseline["model_path"],
        "candidate_model": candidate["model_path"],
        "baseline_count": baseline_summary["count"],
        "candidate_count": candidate_summary["count"],
        "baseline_failed_count": baseline_summary.get("failed_count", 0),
        "candidate_failed_count": candidate_summary.get("failed_count", 0),
        "baseline_mean_weighted_distance": baseline_mean,
        "candidate_mean_weighted_distance": candidate_mean,
        "mean_weighted_distance_delta": candidate_mean - baseline_mean,
        "mean_weighted_distance_improvement": baseline_mean - candidate_mean,
        "baseline_median_weighted_distance": baseline_median,
        "candidate_median_weighted_distance": candidate_median,
        "median_weighted_distance_delta": candidate_median - baseline_median,
        "median_weighted_distance_improvement": baseline_median - candidate_median,
    }


def main() -> int:
    args = parse_args()
    comparison = compare_reports(
        load_report(args.baseline),
        load_report(args.candidate),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=2)
            file.write("\n")
        comparison["output"] = str(output_path)

    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
