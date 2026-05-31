# NeuroWave Naming

## Dataset IDs

Datasets use `dN`.

Examples:

- `d1`: first small generated dataset, 10 seeds.
- `d2`: scaled generated dataset, 500 seeds.
- `d3`: next PyTorch scale-up dataset, 10,000 seeds.
- `d8`: planned 50,000-seed capability dataset.

Dataset files live under:

```text
data/generated/d1/
data/generated/d2/
data/generated/d3/
data/generated/d8/
```

Do not use `v1`, `v2`, or `v2.0` style names for datasets. Reserve `v...` for model versions.

Dataset IDs should not encode model architecture, target mode, loss, or training settings.
A dataset is only the generated source data. The model ID records how that data was used.

## Model IDs

Historical models used integer versions:

```text
vN_<model_type>_<training_size>
```

Examples:

- `v1_sklearn_mlp_10seeds`
- `v2_sklearn_mlp_500seeds`
- `v3_pytorch_cnn_500seeds`
- `v4_pytorch_cnn_10kseeds`

Keep existing integer model IDs unchanged in reports, checkpoints, and progress notes.

Starting with the next model after `v10`, new comparison models use major/minor
experiment versions:

```text
v<major>.<minor>_<model_type>_<target>_<loss>_<size>_<pooling>_<training_size>
```

Use this versioning rule:

- `v2.0`: first model in the next capability series.
- `v2.1`, `v2.2`, ...: ablations or incremental improvements within the same capability series.
- `v3.0`: next major upgrade, such as a materially different architecture, target representation,
  synth parameterization, training objective, evaluation standard, or data-generation regime.

For newer PyTorch capability experiments, use:

```text
v<major>.<minor>_pytorch_cnn_<target>_<loss>_<size>_<pooling>_<training_size>
```

Recommended tokens:

- Target mode:
  - `full`: predicts the full normalized `SynthConfig` vector, including `freq`.
  - `pitchctx`: pitch-conditioned timbre mode; uses exact synthetic `freq` as input context and does not predict `freq`.
- Loss:
  - `flat`: unweighted/default loss.
  - `weighted`: audibility-weighted loss, currently `--loss-preset audibility`.
- Model size:
  - `small`
  - `medium`
  - `large`
- Pooling:
  - `global`: legacy global pooling.
  - `tfpool`: time-frequency pooling, currently `--pooling-mode time_frequency`.

Example historical capability model IDs:

- `v10_pytorch_cnn_pitchctx_weighted_medium_tfpool_50kseeds`

Example next capability model IDs:

- `v2.0_pytorch_cnn_pitchctx_flat_medium_tfpool_50kseeds`
- `v2.1_pytorch_cnn_pitchctx_weighted_medium_tfpool_50kseeds`
- `v2.2_pytorch_cnn_pitchctx_flat_large_tfpool_50kseeds`

Shorter IDs are acceptable while iterating, but serious comparison runs should include
target, loss, size, pooling, and training size so reports are self-explanatory.

Model checkpoints live under:

```text
models/<model_id>.joblib
models/<model_id>.pt
```

## Run Reports

Training reports use:

```text
runs/training/<model_id>_metrics.json
```

Evaluation reports use:

```text
runs/evaluation/<model_id>_on_<dataset_id>_eval.json
```

Comparison reports use:

```text
runs/evaluation/<baseline_model_id>_vs_<candidate_model_id>_on_<dataset_id>.json
```

If a report uses a fixed validation or benchmark subset, keep the dataset ID in the report
name and store the exact indices inside the report JSON rather than encoding index ranges
in the filename.

This keeps dataset identity, model identity, and experiment purpose separate.

## Recommended Current Commands

For the next 50,000-seed capability run:

```text
Dataset: d8
Model: v2.0_pytorch_cnn_pitchctx_flat_medium_tfpool_50kseeds
```

This means:

- Dataset `d8`.
- PyTorch CNN model.
- Pitch-conditioned timbre target mode.
- Flat/default loss.
- Medium model capacity.
- Time-frequency pooling.
- 50,000 generated training examples.
