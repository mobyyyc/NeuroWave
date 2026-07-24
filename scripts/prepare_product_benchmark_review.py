"""Prepare a balanced blinded A/B listening review from two product-benchmark reports."""

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-a", required=True, help="First product-benchmark report JSON.")
    parser.add_argument("--label-a", required=True, help="Private label for the first report.")
    parser.add_argument("--report-b", required=True, help="Second product-benchmark report JSON.")
    parser.add_argument("--label-b", required=True, help="Private label for the second report.")
    parser.add_argument("--output-dir", required=True, help="New or empty directory for the review package.")
    parser.add_argument("--cases-per-category", type=int, default=2)
    return parser.parse_args()


def load_report(path):
    source = Path(path)
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report.get("benchmark_id"), str) or not isinstance(report.get("cases"), list):
        raise ValueError(f"not a product-benchmark report: {source}")
    return report


def select_review_cases(report_a, report_b, cases_per_category):
    """Select the largest model disagreements, balanced across benchmark categories."""
    if cases_per_category < 1:
        raise ValueError("cases_per_category must be at least 1")
    if report_a["benchmark_id"] != report_b["benchmark_id"]:
        raise ValueError("reports must use the same benchmark_id")
    by_id_a = {case["case_id"]: case for case in report_a["cases"] if "comparison" in case}
    by_id_b = {case["case_id"]: case for case in report_b["cases"] if "comparison" in case}
    if set(by_id_a) != set(by_id_b):
        raise ValueError("reports must contain the same successful case ids")

    category_ids = [category["category"] for category in report_a["ranked_failure_groups"]]
    selected = []
    for category in category_ids:
        candidates = [
            case_id
            for case_id, case in by_id_a.items()
            if category in case["categories"]
        ]
        candidates.sort(
            key=lambda case_id: abs(
                by_id_b[case_id]["comparison"]["weighted_distance"]
                - by_id_a[case_id]["comparison"]["weighted_distance"]
            ),
            reverse=True,
        )
        if len(candidates) < cases_per_category:
            raise ValueError(f"category {category} does not have enough successful cases")
        selected.extend(candidates[:cases_per_category])
    return selected


def ensure_empty_directory(path):
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to mix review files into non-empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def artifact_path(report_path, case, name):
    artifact = case.get("artifacts", {}).get(name)
    if not artifact:
        raise ValueError(f"case {case['case_id']} lacks {name} artifact")
    path = Path(report_path).parent / artifact
    if not path.exists():
        raise FileNotFoundError(f"missing review artifact: {path}")
    return path


def write_review_instructions(path, case_count):
    path.write_text(
        "# NeuroWave Product Benchmark: Blind A/B Review\n\n"
        f"Review {case_count} target/option pairs. For every `case-*/` folder:\n\n"
        "1. Listen to `target.wav`, then `option_a.wav` and `option_b.wav`.\n"
        "2. In `scores.csv`, rate each option from 1 (poor) to 5 (excellent) for timbre and envelope.\n"
        "3. Choose `a`, `b`, or `tie` for overall usefulness; add short notes for audible differences.\n"
        "4. Do not open `answer_key.json` until all rows are completed.\n\n"
        "Options are randomized independently per case. This package hides checkpoint labels but does not\n"
        "hide the sound category. The answer key is for post-review aggregation only.\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    report_a_path = Path(args.report_a)
    report_b_path = Path(args.report_b)
    report_a = load_report(report_a_path)
    report_b = load_report(report_b_path)
    selected_ids = select_review_cases(report_a, report_b, args.cases_per_category)
    output_dir = ensure_empty_directory(args.output_dir)
    by_id_a = {case["case_id"]: case for case in report_a["cases"]}
    by_id_b = {case["case_id"]: case for case in report_b["cases"]}
    answer_key = {"benchmark_id": report_a["benchmark_id"], "options": {}}
    score_rows = []
    for number, case_id in enumerate(selected_ids, start=1):
        case_a = by_id_a[case_id]
        case_b = by_id_b[case_id]
        review_id = f"case-{number:02d}"
        review_dir = output_dir / review_id
        review_dir.mkdir()
        shutil.copy2(artifact_path(report_a_path, case_a, "target_audio"), review_dir / "target.wav")
        options = [(args.label_a, case_a), (args.label_b, case_b)]
        random.Random(f"{report_a['benchmark_id']}:{case_id}").shuffle(options)
        for option_name, (label, case) in zip(("a", "b"), options):
            shutil.copy2(
                artifact_path(
                    report_a_path if label == args.label_a else report_b_path,
                    case,
                    "predicted_audio",
                ),
                review_dir / f"option_{option_name}.wav",
            )
        answer_key["options"][review_id] = {
            "case_id": case_id,
            "category": case_a["categories"],
            "a": options[0][0],
            "b": options[1][0],
        }
        score_rows.append(
            {
                "review_id": review_id,
                "category": ",".join(case_a["categories"]),
                "a_timbre_1_to_5": "",
                "b_timbre_1_to_5": "",
                "a_envelope_1_to_5": "",
                "b_envelope_1_to_5": "",
                "overall_preference_a_b_tie": "",
                "notes": "",
            }
        )
    write_review_instructions(output_dir / "README.md", len(selected_ids))
    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=score_rows[0].keys())
        writer.writeheader()
        writer.writerows(score_rows)
    (output_dir / "answer_key.json").write_text(json.dumps(answer_key, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "benchmark_id": report_a["benchmark_id"],
                "review_cases": len(selected_ids),
                "output_dir": str(output_dir),
                "score_sheet": str(output_dir / "scores.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
