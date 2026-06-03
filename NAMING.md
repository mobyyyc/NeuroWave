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

New comparison models use short major/minor experiment names:

```text
v<major>.<minor>_<change>
```

Use this versioning rule:

- `v2.0`: first model in the next capability series.
- `v2.1`, `v2.2`, ...: ablations or incremental improvements within the same capability series.
- `v3.0`: next major upgrade, such as a materially different architecture, target representation,
  synth parameterization, training objective, evaluation standard, or data-generation regime.
- `v3.1`: same v3 model setup scaled to a larger dataset.
- `v3.2`: v3 model setup with oscillator-mix target or diagnostic changes. Use this for
  canonical oscillator representation work where levels are learned by audible waveform
  contribution rather than arbitrary oscillator slot identity.
- `v3.3`: v3 model setup with main/detuned oscillator roles. Use this when the model
  treats known pitch as the main oscillator frequency and learns the second oscillator
  as relative detune plus mix balance.
- `v3.4`: v3.3 model setup with audibility-aware loss. Use this when the target
  representation stays main/detuned but the objective weights oscillator mistakes by
  audible importance.
- `v3.5`: v3.4 model setup with noise-aware detune loss. Use this when detune loss is
  reduced for noise oscillators and audible noise waveform identity is emphasized.

The suffix should describe only what changed or what the experiment proves. Do not encode
the full architecture, loss, target mode, pooling mode, or training configuration in every
model name. Those details belong inside the training report.

Recommended suffix examples:

- `restructure`: architecture/target/loss restructure.
- `500ksamples`: same setup trained on a 500k dataset.
- `oscmix`: canonical oscillator-mix target or diagnostics.
- `main_detuned_mix`: base-frequency oscillator plus relative detuned oscillator target.
- `audible_loss`: audibility-aware loss for waveform, detune, balance, and quiet-level
  overshoot.
- `noise_detune_loss`: audibility-aware loss plus reduced detune pressure for noise
  oscillator targets.
- `losscheck`: focused loss-function ablation.
- `evalfix`: evaluation/reporting-only change.

Example historical capability model IDs:

- `v10_pytorch_cnn_pitchctx_weighted_medium_tfpool_50kseeds`

Example recent and next capability model IDs:

- `v3.0_restructure`
- `v3.1_500ksamples`
- `v3.2_oscmix`
- `v3.3_main_detuned_mix`
- `v3.4_audible_loss`
- `v3.5_noise_detune_loss`

Existing long model IDs should stay unchanged in old reports and checkpoints. New model
IDs should stay short and human-readable.

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

## Product Naming

Desktop app releases should use app-version IDs, separate from model IDs:

```text
app0.1_windows_prototype
app0.2_crop_predict
app1.0_windows
```

App run folders should use a timestamp plus a short input stem:

```text
runs/app/YYYYMMDD_HHMMSS_<input_stem>/
```

Each app run folder should contain:

```text
target_crop.wav
predicted_patch.json
predicted.wav
target_spectrogram.*
predicted_spectrogram.*
summary.json
```

Website releases should use website-version IDs only when needed:

```text
site0.1_landing
site0.2_examples
```

Keep these separate:

- Dataset IDs: `dN`
- Model IDs: `vN_suffix`
- App release IDs: `appN.N_suffix`
- Website release IDs: `siteN.N_suffix`
- App run folders: timestamped local run directories

## Recommended Current Commands

For the next oscillator-mix capability run:

```text
Dataset: d10 or the current 500k dataset
Model: v3.5_noise_detune_loss
```

This means:

- Dataset `d10` or the current 500k generated dataset.
- PyTorch CNN model.
- Pitch-conditioned timbre target mode.
- Main/detuned oscillator-mix representation and diagnostics.
- Audibility-aware loss.
- Noise-aware detune loss.
- Grouped continuous heads.
- Group-balanced loss.
- Large model capacity.
- Time-frequency pooling.
- 500,000 generated training examples.
