# NeuroWave ML Framework Decision

Decision: start Milestone E with `scikit-learn`.

## Why

- `scikit-learn` is already installed in the project environment.
- The current Python version is `3.14.0`; PyTorch and TensorFlow are not installed locally.
- The next useful step is proving the data path:
  `metadata.jsonl -> audio features -> normalized parameter vector`.
- A small regression baseline is faster to build, test, and debug than a neural network stack.

## Initial Model Shape

Use a scikit-learn regressor to predict the normalized `SynthConfig` vector.

Likely first option:

```text
audio features -> MLPRegressor -> normalized synth parameter vector
```

The first feature input can be a compact vector derived from existing feature helpers.
Full CNN-style spectrogram modeling should wait until the baseline loader and evaluation
loop are working.

## Deferred

PyTorch remains the likely next framework when NeuroWave needs:

- CNNs over mel spectrograms.
- GPU acceleration.
- larger datasets.
- checkpointed neural network training.

For now, the goal is a reliable first ML baseline, not maximum model complexity.
