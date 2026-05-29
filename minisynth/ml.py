"""Machine-learning baselines for predicting synth parameters."""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from minisynth.dataset import load_training_dataset


DEFAULT_MLP_HIDDEN_LAYERS = (32,)
DEFAULT_MLP_MAX_ITER = 500
DEFAULT_TEST_SIZE = 0.2


def create_mlp_regressor(
    hidden_layer_sizes=DEFAULT_MLP_HIDDEN_LAYERS,
    random_state=0,
    max_iter=DEFAULT_MLP_MAX_ITER,
):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
        ),
    )


def train_mlp_regressor(features, targets, **kwargs):
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)

    if x.ndim != 2:
        raise ValueError("features must be a 2D array")

    if y.ndim != 2:
        raise ValueError("targets must be a 2D array")

    if len(x) != len(y):
        raise ValueError("features and targets must have the same number of rows")

    model = create_mlp_regressor(**kwargs)
    model.fit(x, y)
    return model


def predict_parameter_vectors(model, features):
    predictions = model.predict(np.asarray(features, dtype=float))
    return np.clip(predictions, 0.0, 1.0)


def parameter_mae(targets, predictions):
    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must have the same shape")

    return float(mean_absolute_error(y_true, y_pred))


def train_mlp_from_metadata(
    metadata_path,
    hidden_layer_sizes=DEFAULT_MLP_HIDDEN_LAYERS,
    max_iter=DEFAULT_MLP_MAX_ITER,
    random_state=0,
    test_size=DEFAULT_TEST_SIZE,
):
    features, targets = load_training_dataset(metadata_path)

    if len(features) < 2:
        raise ValueError("at least 2 training rows are required")

    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
    )
    model = train_mlp_regressor(
        train_features,
        train_targets,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )

    train_predictions = predict_parameter_vectors(model, train_features)
    test_predictions = predict_parameter_vectors(model, test_features)

    return {
        "model": model,
        "metrics": {
            "metadata_path": str(metadata_path),
            "num_samples": int(len(features)),
            "num_features": int(features.shape[1]),
            "num_targets": int(targets.shape[1]),
            "train_samples": int(len(train_features)),
            "test_samples": int(len(test_features)),
            "train_mae": parameter_mae(train_targets, train_predictions),
            "test_mae": parameter_mae(test_targets, test_predictions),
        },
    }
