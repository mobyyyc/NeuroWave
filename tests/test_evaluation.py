import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.exceptions import ConvergenceWarning

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.dataset import write_random_dataset_files
from minisynth.engine import render_patch
from minisynth.evaluation import evaluate_audio_prediction
from minisynth.evaluation import summarize_weighted_distances
from minisynth.ml import train_mlp_from_metadata


class TestPredictionEvaluation(unittest.TestCase):
    def test_evaluate_audio_prediction_returns_patch_audio_and_comparison(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=4000,
                count=5,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                training = train_mlp_from_metadata(
                    metadata_path,
                    hidden_layer_sizes=(4,),
                    max_iter=10,
                    random_state=1,
                    test_size=0.4,
                )

        target_audio = render_patch(length=1.0)
        result = evaluate_audio_prediction(
            training["model"],
            target_audio,
            DEFAULT_SAMPLE_RATE,
            refine_iterations=2,
            refine_seed=1,
            refine_step_size=0.01,
        )

        self.assertIn("freq", result["patch"])
        self.assertEqual(result["rendered_audio"].ndim, 1)
        self.assertIn("weighted_distance", result["comparison"])
        self.assertGreaterEqual(result["comparison"]["weighted_distance"], 0.0)
        self.assertEqual(result["refinement"]["iterations"], 2)
        self.assertLessEqual(
            result["refinement"]["best_score"],
            result["refinement"]["initial_score"],
        )

    def test_summarize_weighted_distances_returns_aggregate_scores(self):
        summary = summarize_weighted_distances(
            [
                {"comparison": {"weighted_distance": 3.0}},
                {"comparison": {"weighted_distance": 1.0}},
                {"comparison": {"weighted_distance": 2.0}},
            ]
        )

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean_weighted_distance"], 2.0)
        self.assertEqual(summary["median_weighted_distance"], 2.0)
        self.assertEqual(summary["min_weighted_distance"], 1.0)
        self.assertEqual(summary["max_weighted_distance"], 3.0)

    def test_summarize_weighted_distances_rejects_empty_results(self):
        with self.assertRaises(ValueError):
            summarize_weighted_distances([])


if __name__ == "__main__":
    unittest.main()
