import json
from pathlib import Path
import unittest

from minisynth.product_benchmark import load_product_benchmark, validate_product_benchmark
from scripts.evaluate_product_benchmark import category_summaries, ranked_failure_groups
from scripts.prepare_product_benchmark_review import select_review_cases
from scripts.summarize_product_benchmark_review import summarize_review


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "product_benchmark_v1.json"


class TestProductBenchmark(unittest.TestCase):
    def test_committed_manifest_is_valid_and_balanced(self):
        benchmark = load_product_benchmark(MANIFEST)

        self.assertEqual(benchmark["id"], "neurowave_product_benchmark_v1")
        self.assertEqual(len(benchmark["cases"]), 36)
        self.assertEqual(benchmark["expected_cases_per_category"], 6)

    def test_rejects_duplicate_source_seed(self):
        benchmark = json.loads(MANIFEST.read_text(encoding="utf-8"))
        benchmark["cases"][1]["source"]["seed"] = benchmark["cases"][0]["source"]["seed"]
        benchmark["cases"][1]["source"]["index"] = benchmark["cases"][0]["source"]["index"]

        with self.assertRaisesRegex(ValueError, "reuse source seeds"):
            validate_product_benchmark(benchmark)

    def test_rejects_non_benchmark_source_partition(self):
        benchmark = json.loads(MANIFEST.read_text(encoding="utf-8"))
        benchmark["source_dataset"]["partition"] = "dev"

        with self.assertRaisesRegex(ValueError, "immutable benchmark partition"):
            validate_product_benchmark(benchmark)

    def test_category_summaries_and_failure_ranking(self):
        results = [
            {
                "case_id": "one",
                "categories": ["noise"],
                "comparison": {"weighted_distance": 4.0},
            },
            {
                "case_id": "two",
                "categories": ["noise"],
                "comparison": {"weighted_distance": 2.0},
            },
            {"case_id": "three", "categories": ["detune"], "error": "render failed"},
        ]

        summaries = category_summaries(results, ["noise", "detune"])

        self.assertEqual(summaries["noise"]["mean_weighted_distance"], 3.0)
        self.assertEqual(summaries["noise"]["worst_case_ids"], ["one", "two"])
        self.assertEqual(summaries["detune"]["failed_count"], 1)
        self.assertEqual(ranked_failure_groups(summaries)[0]["category"], "noise")

    def test_review_selection_balances_categories_by_model_disagreement(self):
        report_a = {
            "benchmark_id": "example",
            "ranked_failure_groups": [{"category": "noise"}, {"category": "detune"}],
            "cases": [
                {"case_id": "noise-low", "categories": ["noise"], "comparison": {"weighted_distance": 1}},
                {"case_id": "noise-high", "categories": ["noise"], "comparison": {"weighted_distance": 2}},
                {"case_id": "detune-low", "categories": ["detune"], "comparison": {"weighted_distance": 3}},
                {"case_id": "detune-high", "categories": ["detune"], "comparison": {"weighted_distance": 4}},
            ],
        }
        report_b = {
            **report_a,
            "cases": [
                {**case, "comparison": {"weighted_distance": case["comparison"]["weighted_distance"] + delta}}
                for case, delta in zip(report_a["cases"], (1, 10, 2, 9))
            ],
        }

        selected = select_review_cases(report_a, report_b, cases_per_category=1)

        self.assertEqual(selected, ["noise-high", "detune-high"])

    def test_review_summary_unblinds_scores_and_preferences(self):
        rows = [
            {
                "review_id": "case-01",
                "a_timbre_1_to_5": "4",
                "b_timbre_1_to_5": "5",
                "a_envelope_1_to_5": "3",
                "b_envelope_1_to_5": "5",
                "overall_preference_a_b_tie": "b",
                "notes": "b is closer",
            }
        ]
        answer_key = {
            "benchmark_id": "example",
            "options": {
                "case-01": {"case_id": "noise-001", "category": ["audible_noise"], "a": "v3.5", "b": "v3.4"}
            },
        }

        summary = summarize_review(rows, answer_key)

        self.assertEqual(summary["preference_counts"]["v3.4"], 1)
        self.assertEqual(summary["models"]["v3.5"]["mean_timbre"], 4.0)
        self.assertEqual(summary["models"]["v3.4"]["mean_envelope"], 5.0)
