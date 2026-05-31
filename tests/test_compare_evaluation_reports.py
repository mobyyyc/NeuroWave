import unittest

from scripts.compare_evaluation_reports import compare_reports


class TestCompareEvaluationReports(unittest.TestCase):
    def test_compare_reports_calculates_weighted_distance_improvement(self):
        baseline = {
            "model_path": "models/v1_sklearn_mlp_10seeds.joblib",
            "summary": {
                "count": 20,
                "failed_count": 2,
                "mean_weighted_distance": 100.0,
                "median_weighted_distance": 80.0,
            },
        }
        candidate = {
            "model_path": "models/v2_sklearn_mlp_500seeds.joblib",
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

    def test_compare_reports_includes_parameter_and_waveform_metrics(self):
        baseline = {
            "model_path": "models/v8_pytorch_cnn_50kseeds.pt",
            "model_metrics": {
                "test_loss": 0.04,
                "test_mae": 0.13,
                "test_continuous_mae": 0.12,
                "test_waveform_accuracy": 0.5,
                "test_per_parameter_mae": {
                    "freq": 0.2,
                    "cutoff": 0.1,
                },
                "test_grouped_mae": {
                    "adsr": 0.12,
                    "filter": 0.1,
                },
                "test_waveform_accuracy_by_name": {
                    "osc1_wave": 0.4,
                },
            },
            "summary": {
                "count": 20,
                "failed_count": 0,
                "mean_weighted_distance": 100.0,
                "median_weighted_distance": 80.0,
            },
        }
        candidate = {
            "model_path": "models/v9_pytorch_cnn_200kseeds.pt",
            "model_metrics": {
                "test_loss": 0.03,
                "test_mae": 0.11,
                "test_continuous_mae": 0.1,
                "test_waveform_accuracy": 0.75,
                "test_per_parameter_mae": {
                    "freq": 0.15,
                    "cutoff": 0.08,
                },
                "test_grouped_mae": {
                    "adsr": 0.1,
                    "filter": 0.08,
                },
                "test_waveform_accuracy_by_name": {
                    "osc1_wave": 0.6,
                },
            },
            "summary": {
                "count": 20,
                "failed_count": 0,
                "mean_weighted_distance": 90.0,
                "median_weighted_distance": 70.0,
            },
        }

        comparison = compare_reports(baseline, candidate)

        self.assertAlmostEqual(comparison["test_mae_delta"], -0.02)
        self.assertAlmostEqual(comparison["test_waveform_accuracy_delta"], 0.25)
        self.assertAlmostEqual(
            comparison["test_per_parameter_mae_delta_by_name"]["cutoff"],
            -0.02,
        )
        self.assertAlmostEqual(
            comparison["test_grouped_mae_delta_by_name"]["adsr"],
            -0.02,
        )
        self.assertAlmostEqual(
            comparison["test_waveform_accuracy_by_name_delta_by_name"]["osc1_wave"],
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
