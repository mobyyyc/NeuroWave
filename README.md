# NeuroWave

NeuroWave is a Python synthesizer project being built toward machine-learning-driven sound recreation.

The current system renders known synth parameters into audio, generates synthetic datasets, trains inverse models from audio features, and evaluates predicted synth patches by rendering them back to audio.

Architecture note: the repository and Python package still use the historical `MiniSynth` / `minisynth` names internally. Do not rename folders, imports, or package paths unless that migration is planned separately.

## Project Map

- `minisynth/`: synth engine, schema, audio features, dataset helpers, and ML model code.
- `presets/`: human-readable synth presets.
- `scripts/`: commands for rendering, dataset generation, training, prediction, and evaluation.
- `data/generated/`: local generated datasets. Ignored by git.
- `models/`: local trained checkpoints. Ignored by git.
- `runs/`: local training, prediction, and evaluation reports. Ignored by git.
- `PLAN.md`: long-term roadmap.
- `PROGRESS.md`: daily task tracker.
- `NAMING.md`: dataset, model, and report naming rules.

## Environment Setup

Create and activate the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the normal project dependencies:

```bash
python -m pip install -r requirements.txt
```

For local macOS or CPU-only development, install PyTorch normally:

```bash
python -m pip install torch
```

For NVIDIA CUDA training, install CUDA-enabled PyTorch after the base requirements:

```bash
python -m pip install -r requirements-cuda.txt
```

Verify which training device PyTorch can use:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('mps:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

The training scripts automatically prefer CUDA, then Apple MPS, then CPU. You can override that with `--device cuda`, `--device mps`, or `--device cpu`.

## Render A Preset

Render the current `dark_saw` preset:

```bash
python scripts/render_patch.py presets/dark_saw.json dark_saw.wav
```

Run the smoke render:

```bash
python scripts/smoke_render.py
```

## Generate Datasets

Datasets are named `dN`. Historical models used long descriptive IDs, but new
models use short change-focused IDs such as `v3.0_restructure`,
`v3.1_500ksamples`, and `v3.2_oscmix`. Keep dataset IDs and model IDs separate.

Generate the first tiny local dataset:

```bash
python scripts/random_patch.py --dataset-version d1 --seed 1000 --count 10 --workers 1
```

Generate `d2`, the 500-seed dataset used for baseline PyTorch training:

```bash
python scripts/random_patch.py --dataset-version d2 --seed 2000 --count 500 --workers 0
python scripts/export_mel_tensors.py --dataset-version d2 --workers 0
```

Generate a larger future dataset, such as `d3` with 10,000 seeds:

```bash
python scripts/random_patch.py --dataset-version d3 --seed 3000 --count 10000 --workers 0
python scripts/export_mel_tensors.py --dataset-version d3 --workers 0
```

Worker rules:

- `--workers 1`: serial mode.
- `--workers 0`: conservative automatic multicore mode.
- `--workers N`: manually use `N` worker processes.

The dataset code limits NumPy/SciPy worker threads so multicore generation does not overload the CPU with nested thread pools.

## Train PyTorch Models

Use PyTorch for future model-quality work. The scikit-learn model remains only as a lightweight baseline.

Future training uses the current best v3 defaults: pitch-conditioned timbre,
waveform classification, large CNN, time-frequency pooling, grouped heads,
group-balanced loss, AdamW, step LR, early stopping, and best-validation
checkpoint saving. These are report fields, not model-name tokens.

Template for future models:

```bash
python scripts/train_torch.py \
  --model-id v3.2_oscmix \
  --tensor-data data/generated/dN/features \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --model-output models/v3.2_oscmix.pt \
  --metrics-output runs/training/v3.2_oscmix_metrics.json
```

Training output:

- Checkpoint: `models/<model_id>.pt`
- Metrics report: `runs/training/<model_id>_metrics.json`
- Console progress: device selection, epochs, batches, and final metrics unless `--quiet` is used.
- Metrics include train/test loss, train/test MAE, continuous-parameter MAE,
  per-parameter MAE, grouped MAE, waveform accuracy, and compact loss-history tails.

The current best proven model family is the v3 pitch-conditioned grouped-head setup. It keeps exact synthetic pitch as context, uses separate heads for duration, oscillator, filter, and ADSR controls, and trains with group-balanced loss so one parameter group cannot hide another.

`v3.1` scaled that setup to 500k synthetic examples and became the current best checkpoint. The next planned improvement is `v3.2`, focused on oscillator-mix representation. The two oscillator slots are partly exchangeable: a quiet saw in oscillator 1 plus a loud sine in oscillator 2 can be equivalent, or nearly equivalent, to the same saw/sine level contributions assigned to the opposite slots. Future v3.2 work should therefore canonicalize oscillator targets or predict total oscillator level plus balance/per-wave level contribution, so the model learns the audible mix instead of arbitrary slot identity.

The v3 defaults are intentionally not exposed as routine CLI switches. If a future
experiment needs a different architecture or loss, change the code deliberately and
record that change as the model suffix.

## Evaluate PyTorch Models

Evaluate a current PyTorch checkpoint on a generated dataset:

```bash
python scripts/evaluate_dataset_torch.py \
  --metadata data/generated/dN/metadata.jsonl \
  --model models/v3.2_oscmix.pt \
  --count 1000 \
  --start-index 0 \
  --device cuda \
  --output runs/evaluation/v3.2_oscmix_on_dN_eval.json
```

Use a smaller count for quick checks:

```bash
python scripts/evaluate_dataset_torch.py \
  --metadata data/generated/dN/metadata.jsonl \
  --model models/v3.2_oscmix.pt \
  --count 200 \
  --output runs/evaluation/v3.2_oscmix_smoke_eval.json
```

Evaluation reports are compact by default: they include weighted audio distance summaries,
compact checkpoint metrics, prediction-distribution diagnostics, and a
`diagnostics.worst_clips` section ranked by weighted audio distance with the largest
parameter errors. Use `--diagnostics-top-n 20` for a larger worst-case review set,
`--include-clips` for compact per-clip rows, or `--include-full-clips` when you need
full target patches, predicted patches, and normalized per-parameter errors for
debugging.

## Pitch And Length Strategy

For future model-quality work, `freq` should be treated as pitch context rather than a core timbre prediction target. Synthetic datasets already know the exact patch `freq`, so the model can use that value to interpret the mel spectrogram without spending output capacity predicting pitch.

For real one-note clips later, estimate pitch before prediction with a classical monophonic pitch estimator or allow manual pitch input.

Keep `length` visible to the model design for now because duration affects ADSR interpretation and whether a sound behaves like a pluck, key, or pad.

## Predict One Patch

Predict a patch JSON from one audio clip using a PyTorch checkpoint:

```bash
python scripts/predict_patch_torch.py \
  data/generated/dN/audio/patch_000000_seed_0000.wav \
  runs/pytorch_prediction/v3.2_oscmix_patch.json \
  --model models/v3.2_oscmix.pt \
  --freq 440
```

Render and compare a PyTorch prediction against the target audio:

```bash
python scripts/evaluate_prediction_torch.py \
  data/generated/dN/audio/patch_000000_seed_0000.wav \
  --model models/v3.2_oscmix.pt \
  --freq 440 \
  --output-dir runs/pytorch_prediction/v3.2_oscmix_patch_eval
```

## Compare Reports

Compare two evaluation reports:

```bash
python scripts/compare_evaluation_reports.py \
  runs/evaluation/v2_sklearn_mlp_500seeds_on_d2_eval.json \
  runs/evaluation/v3_pytorch_cnn_500seeds_on_d2_eval.json \
  --output runs/evaluation/v2_sklearn_mlp_500seeds_vs_v3_pytorch_cnn_500seeds_on_d2.json
```

## Legacy Scikit-Learn Baseline

Train the old MLP baseline only when you need a quick pipeline sanity check:

```bash
python scripts/train_mlp.py \
  --metadata data/generated/d2/metadata.jsonl \
  --model-output models/v2_sklearn_mlp_500seeds.joblib \
  --metrics-output runs/training/v2_sklearn_mlp_500seeds_metrics.json
```

Evaluate it:

```bash
python scripts/evaluate_dataset.py \
  --metadata data/generated/d2/metadata.jsonl \
  --model models/v2_sklearn_mlp_500seeds.joblib \
  --count 20 \
  --output runs/evaluation/v2_sklearn_mlp_500seeds_on_d2_eval.json
```

## Tests

Run the unit tests:

```bash
python -m unittest discover -s tests
```

Run the smoke render:

```bash
python scripts/smoke_render.py
```
