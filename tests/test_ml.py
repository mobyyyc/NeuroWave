import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from minisynth.ml import (
    create_mlp_regressor,
    predict_parameter_vectors,
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


if __name__ == "__main__":
    unittest.main()
