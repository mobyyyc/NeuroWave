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

## Milestone F Review

Decision: keep the scikit-learn MLP as the baseline, but move the next model-development
track toward PyTorch and spectrogram-based learning.

Evidence from scaled synthetic training:

- The v2 dataset has 500 examples.
- The v2 MLP checkpoint reached `train_mae = 0.2052` and `test_mae = 0.2089`.
- On the same 20 v2 clips, the v2 model improved mean weighted audio distance from
  `328.03` to `150.61` compared with the 10-example v1 tiny model.
- Median weighted audio distance improved from `157.55` to `71.80`.
- Both models still had one failed predicted render in the 20-clip comparison.

Interpretation:

- More synthetic data helps, so the pipeline is learning something real.
- The current compact 7-number feature vector is too weak for the final inverse synth goal.
- The MLP is useful as a regression baseline and regression-test target.
- The next serious model should consume richer audio features, likely mel spectrograms,
  which makes PyTorch the right next framework.

Near-term rule:

- Keep scikit-learn commands working for quick sanity checks.
- Do not spend major effort tuning the current MLP.
- Add PyTorch only when the project is ready to train a spectrogram model and the
  dependency/runtime decision is explicit.
