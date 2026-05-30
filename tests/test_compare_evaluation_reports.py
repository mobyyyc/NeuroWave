import unittest

from scripts.compare_evaluation_reports import compare_reports


class TestCompareEvaluationReports(unittest.TestCase):
    def test_compare_reports_calculates_weighted_distance_improvement(self):
        baseline = {
            "model_path": "models/v1.joblib",
            "summary": {
                "count": 20,
                "failed_count": 2,
                "mean_weighted_distance": 100.0,
                "median_weighted_distance": 80.0,
            },
        }
        candidate = {
            "model_path": "models/v2.joblib",
            "summary": {
                "count": 20,
                "failed_count": 1,
                "mean_weighted_distance": 75.0,
                "median_weighted_distance": 60.0,
            },
        }

        comparison = compare_reports(baseline, candidate)

        self.assertEqual(comparison["mean_weighted_distance_delta"], -25.0)
        self.assertEqual(comparison["mean_weighted_distance_improvement"], 25.0)
        self.assertEqual(comparison["median_weighted_distance_delta"], -20.0)
        self.assertEqual(comparison["candidate_failed_count"], 1)


if __name__ == "__main__":
    unittest.main()
