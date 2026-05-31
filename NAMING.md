# NeuroWave Naming

## Dataset IDs

Datasets use `dN`.

Examples:

- `d1`: first small generated dataset, 10 seeds.
- `d2`: scaled generated dataset, 500 seeds.

Dataset files live under:

```text
data/generated/d1/
data/generated/d2/
```

Do not use `v1` or `v2` for datasets. Reserve `vN` for model versions.

## Model IDs

Models use:

```text
vN_<model_type>_<training_size>
```

Examples:

- `v1_sklearn_mlp_10seeds`
- `v2_sklearn_mlp_500seeds`
- `v3_pytorch_cnn_10kseeds`

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

This keeps dataset identity, model identity, and experiment purpose separate.
