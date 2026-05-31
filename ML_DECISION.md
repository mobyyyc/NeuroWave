# NeuroWave ML Framework Decision

Decision: start Milestone E with `scikit-learn`.

## Why

- `scikit-learn` is already installed in the project environment.
- At the time, PyTorch and TensorFlow were not installed locally.
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

- The `d2` dataset has 500 examples.
- The `v2_sklearn_mlp_500seeds` checkpoint reached `train_mae = 0.2052` and `test_mae = 0.2089`.
- On the same 20 `d2` clips, `v2_sklearn_mlp_500seeds` improved mean weighted audio distance from
  `328.03` to `150.61` compared with `v1_sklearn_mlp_10seeds`.
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

## Milestone G Review

Decision: keep scikit-learn as a lightweight baseline, but make PyTorch the primary
model-development path.

Evidence from the first PyTorch spectrogram model:

- `v3_pytorch_cnn_500seeds` trained successfully on exported `d2` mel tensors.
- The PyTorch checkpoint predicts valid patches, renders audio, and evaluates across
  synthetic dataset clips.
- On the same `d2` evaluation setup, `v3_pytorch_cnn_500seeds` improved mean weighted
  audio distance from `150.61` to `132.54` compared with `v2_sklearn_mlp_500seeds`.
- Median weighted audio distance improved from `71.80` to `57.73`.
- The PyTorch model had zero failed predictions in the 20-clip report; the sklearn
  baseline had one failed prediction.

Interpretation:

- PyTorch has passed the threshold for becoming the serious model path.
- The sklearn baseline is still useful because it is fast, simple, and catches data
  pipeline regressions without requiring neural-network training.
- Removing sklearn now would save little complexity and would remove a useful reference
  point while the PyTorch model is still early.

Current rule:

- Keep sklearn code and commands working as a quick regression baseline.
- Do not tune sklearn unless it is needed to debug data or evaluation behavior.
- Use PyTorch for future model quality work, larger datasets, spectrogram learning,
  checkpointing, and eventual real-audio approximation.

## 10k Seed Scale-Up Review

Decision: do not assume more seeds alone will solve accuracy.

Evidence from `v4_pytorch_cnn_10kseeds`:

- Trained on `d3`, a 10,000-seed generated dataset.
- Parameter prediction improved versus `v3`: `test_mae` moved from about `0.1924`
  to `0.1652`.
- Weighted audio distance did not improve on the comparable `d2` 20-clip report:
  `v3` mean was `132.54`, while `v4` mean was `202.85`.
- `v4` on a 200-clip `d3` report produced mean weighted distance `150.01` and
  median `64.62` with zero failures.

Interpretation:

- The model is learning parameter averages better, but that does not yet translate into
  better rendered audio.
- The current plain parameter MSE training objective is probably too weak for perceptual
  synth matching.
- The next quality improvement should focus on evaluation-informed training, parameter
  weighting, better splits, or architecture changes before scaling to 50k+ seeds.

## Model Capability Roadmap Decision

Decision: add a dedicated model-capability milestone before real-audio work.

Evidence from later local PyTorch reports:

- `v5_pytorch_cnn_10kseeds` reached `test_loss = 0.0461` and `test_mae = 0.1616`.
- `v6_pytorch_cnn_50kseeds` reached `test_loss = 0.0406` and `test_mae = 0.1446`.
- `v7_pytorch_cnn_50kseeds` reached `test_loss = 0.0377` and `test_mae = 0.1357`.
- `v8_pytorch_cnn_50kseeds` reached `test_loss = 0.0375` and `test_mae = 0.1339`.
- `v9_pytorch_cnn_200kseeds` reached `test_loss = 0.0356` and `test_mae = 0.1284`.

Interpretation:

- The current normalized parameter MSE is already below `0.1`, but the average normalized
  parameter MAE is still far from the desired `0.05` target.
- Train and test metrics remain close across the larger runs, so the current pattern does
  not look like simple overfitting.
- The current model is probably underpowered for the target: it is a small CNN with a
  single averaged regression head and early global pooling.
- The current target representation is also likely limiting quality because waveform
  categories are encoded as scalar regression values.
- Future quality work should make the model and target design strong enough that remaining
  errors can be blamed on data quality, synth ambiguity, or representation limits rather
  than an obviously weak network.

Current rule:

- Do not move straight to real-audio approximation until synthetic model capability is
  measured and improved.
- Add per-parameter metrics before changing the model so every later experiment explains
  what improved and what did not.
- Treat waveform identity, continuous timbre parameters, pitch context, and duration context
  as separable modeling concerns unless evidence shows a single regression vector is sufficient.
- Use stronger PyTorch architectures, better checkpoint selection, optimizer controls, and
  fixed benchmark reports before scaling datasets again.

## Pitch And Length Target Decision

Decision: do not make `freq` a core timbre prediction target in the next model design.
Use exact synthetic patch `freq` as pitch conditioning during synthetic training, and
use classical pitch estimation or manual pitch input later for real single-note clips.

Reason:

- Different fundamentals produce different mel spectrograms for the same synth settings, so
  the model should be pitch-aware.
- Synthetic datasets already contain exact `freq`, so forcing the model to predict it wastes
  output capacity and can contaminate timbre-quality metrics.
- For real monophonic clips, pitch is a well-scoped preprocessing problem that can start with
  YIN or pYIN before introducing any learned pitch model.

Decision: keep `length` visible to the model design for now.

Reason:

- Duration changes how ADSR parameters are interpreted.
- The difference between a pluck, key-like sound, and pad depends partly on note length and
  envelope timing.
- Removing `length` too early could make envelope prediction look worse for the wrong reason.

Near-term rule:

- Add grouped target metrics before changing the training target: timbre metrics excluding
  `freq`, ADSR metrics, oscillator metrics, filter metrics, and waveform accuracy.
- Then train a pitch-conditioned timbre model that receives exact synthetic `freq` as input
  context but does not include `freq` in the output loss.
- Keep real-audio pitch estimation in Milestone I unless synthetic pitch conditioning exposes
  a blocker earlier.
