"""Validate and unblind a completed NeuroWave product-benchmark listening review."""

import argparse
import csv
import json
from pathlib import Path


SCORE_FIELDS = (
    "a_timbre_1_to_5",
    "b_timbre_1_to_5",
    "a_envelope_1_to_5",
    "b_envelope_1_to_5",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Completed blinded scores CSV.")
    parser.add_argument("--answer-key", required=True, help="Matching answer_key.json.")
    parser.add_argument("--output", help="Summary JSON path; defaults beside the score sheet.")
    return parser.parse_args()


def load_scores(path):
    source = Path(path)
    text = None
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            text = source.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"could not decode score sheet: {source}")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("scores CSV must contain at least one review row")
    return rows


def parse_score(row, field):
    try:
        score = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row.get('review_id', '<unknown>')} requires integer {field}") from error
    if score < 1 or score > 5:
        raise ValueError(f"{row['review_id']} {field} must be in 1..5")
    return score


def summarize_review(rows, answer_key):
    options = answer_key.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError("answer key requires non-empty options")
    row_ids = {row.get("review_id") for row in rows}
    if row_ids != set(options):
        raise ValueError("scores CSV review ids must exactly match answer key review ids")

    models = {}
    categories = {}
    review_rows = []
    preference_counts = {"tie": 0}
    for row in rows:
        review_id = row["review_id"]
        key = options[review_id]
        option_models = {"a": key["a"], "b": key["b"]}
        preference = row.get("overall_preference_a_b_tie", "").strip().lower()
        if preference not in ("a", "b", "tie"):
            raise ValueError(f"{review_id} preference must be a, b, or tie")
        if preference == "tie":
            preference_counts["tie"] += 1
        else:
            preference_counts[option_models[preference]] = preference_counts.get(option_models[preference], 0) + 1

        per_case = {
            "review_id": review_id,
            "case_id": key["case_id"],
            "categories": key["category"],
            "preference": preference,
            "notes": row.get("notes", ""),
            "options": option_models,
        }
        for option in ("a", "b"):
            model = option_models[option]
            model_metrics = models.setdefault(
                model,
                {"case_count": 0, "timbre_total": 0, "envelope_total": 0, "preference_wins": 0},
            )
            timbre = parse_score(row, f"{option}_timbre_1_to_5")
            envelope = parse_score(row, f"{option}_envelope_1_to_5")
            model_metrics["case_count"] += 1
            model_metrics["timbre_total"] += timbre
            model_metrics["envelope_total"] += envelope
            per_case[f"{option}_timbre"] = timbre
            per_case[f"{option}_envelope"] = envelope
            for category in key["category"]:
                category_metrics = categories.setdefault(category, {}).setdefault(
                    model,
                    {"case_count": 0, "timbre_total": 0, "envelope_total": 0, "preference_wins": 0},
                )
                category_metrics["case_count"] += 1
                category_metrics["timbre_total"] += timbre
                category_metrics["envelope_total"] += envelope
        if preference in option_models:
            models[option_models[preference]]["preference_wins"] += 1
            for category in key["category"]:
                categories[category][option_models[preference]]["preference_wins"] += 1
        review_rows.append(per_case)

    def finalize(metrics):
        return {
            "case_count": metrics["case_count"],
            "mean_timbre": metrics["timbre_total"] / metrics["case_count"],
            "mean_envelope": metrics["envelope_total"] / metrics["case_count"],
            "preference_wins": metrics["preference_wins"],
        }

    return {
        "benchmark_id": answer_key.get("benchmark_id", "unknown"),
        "review_case_count": len(rows),
        "preference_counts": preference_counts,
        "models": {model: finalize(metrics) for model, metrics in models.items()},
        "categories": {
            category: {model: finalize(metrics) for model, metrics in values.items()}
            for category, values in categories.items()
        },
        "cases": review_rows,
    }


def main():
    args = parse_args()
    scores_path = Path(args.scores)
    answer_key = json.loads(Path(args.answer_key).read_text(encoding="utf-8"))
    summary = summarize_review(load_scores(scores_path), answer_key)
    output_path = Path(args.output) if args.output else scores_path.with_name("review_summary.json")
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
