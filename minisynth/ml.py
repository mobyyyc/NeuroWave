"""Machine-learning baselines for predicting synth parameters."""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MLP_HIDDEN_LAYERS = (32,)
DEFAULT_MLP_MAX_ITER = 500


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
