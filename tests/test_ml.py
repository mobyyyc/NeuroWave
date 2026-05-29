import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from minisynth.dataset import write_random_dataset_files
from minisynth.ml import (
    create_mlp_regressor,
    parameter_mae,
    predict_parameter_vectors,
    train_mlp_from_metadata,
    train_mlp_regressor,
)


class TestMLBaseline(unittest.TestCase):
    def test_create_mlp_regressor_returns_pipeline(self):
        model = create_mlp_regressor(max_iter=10)

        self.assertIsInstance(model, Pipeline)

    def test_train_mlp_regressor_fits_multi_output_targets(self):
        features = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        targets = np.array(
            [
                [0.0, 0.0],
                [0.25, 0.5],
                [0.5, 0.25],
                [1.0, 1.0],
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = train_mlp_regressor(
                features,
                targets,
                hidden_layer_sizes=(4,),
                max_iter=50,
                random_state=1,
            )
        predictions = predict_parameter_vectors(model, features)

        self.assertEqual(predictions.shape, targets.shape)
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))

    def test_train_mlp_regressor_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            train_mlp_regressor(np.ones(3), np.ones((3, 1)))

        with self.assertRaises(ValueError):
            train_mlp_regressor(np.ones((3, 1)), np.ones(3))

        with self.assertRaises(ValueError):
            train_mlp_regressor(np.ones((3, 1)), np.ones((2, 1)))

    def test_parameter_mae_returns_single_distance(self):
        targets = np.array([[0.0, 0.5], [1.0, 0.25]])
        predictions = np.array([[0.25, 0.5], [0.5, 0.75]])

        self.assertAlmostEqual(parameter_mae(targets, predictions), 0.3125)

    def test_parameter_mae_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            parameter_mae(np.ones((2, 1)), np.ones((2, 2)))

    def test_train_mlp_from_metadata_returns_metrics(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            write_random_dataset_files(
                param_dir=root / "params",
                audio_dir=root / "audio",
                metadata_path=metadata_path,
                seed=2000,
                count=5,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = train_mlp_from_metadata(
                    metadata_path,
                    hidden_layer_sizes=(4,),
                    max_iter=10,
                    random_state=1,
                    test_size=0.4,
                )

        metrics = result["metrics"]

        self.assertIn("model", result)
        self.assertEqual(metrics["num_samples"], 5)
        self.assertEqual(metrics["num_features"], 7)
        self.assertEqual(metrics["num_targets"], 13)
        self.assertEqual(metrics["train_samples"], 3)
        self.assertEqual(metrics["test_samples"], 2)
        self.assertGreaterEqual(metrics["train_mae"], 0.0)
        self.assertGreaterEqual(metrics["test_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
