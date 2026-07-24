import json
from pathlib import Path
import unittest

from minisynth.product_benchmark import load_product_benchmark, validate_product_benchmark
from scripts.evaluate_product_benchmark import category_summaries, ranked_failure_groups


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
