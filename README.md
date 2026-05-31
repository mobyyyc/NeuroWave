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

Datasets are named `dN`. Models are named `vN_<model_type>_<training_size>`. Keep those separate.

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

Train the explicit `d2` PyTorch model:

```bash
python scripts/train_torch.py \
  --model-id v3_pytorch_cnn_500seeds \
  --tensor-data data/generated/d2/features/mel_tensors.npz \
  --model-output models/v3_pytorch_cnn_500seeds.pt \
  --metrics-output runs/training/v3_pytorch_cnn_500seeds_metrics.json
```

Train the current 10,000-seed PyTorch model on `d3`:

```bash
python scripts/train_torch.py \
  --model-id v4_pytorch_cnn_10kseeds \
  --tensor-data data/generated/d3/features/mel_tensors.npz \
  --model-output models/v4_pytorch_cnn_10kseeds.pt \
  --metrics-output runs/training/v4_pytorch_cnn_10kseeds_metrics.json
```

Template for future models:

```bash
python scripts/train_torch.py \
  --model-id vN_pytorch_cnn_<training_size> \
  --tensor-data data/generated/dN/features/mel_tensors.npz \
  --model-output models/vN_pytorch_cnn_<training_size>.pt \
  --metrics-output runs/training/vN_pytorch_cnn_<training_size>_metrics.json
```

Reserve a fixed benchmark subset when you want metrics beyond the validation split:

```bash
python scripts/train_torch.py \
  --model-id vN_pytorch_cnn_<training_size> \
  --tensor-data data/generated/dN/features/mel_tensors.npz \
  --benchmark-size 0.1 \
  --model-output models/vN_pytorch_cnn_<training_size>.pt \
  --metrics-output runs/training/vN_pytorch_cnn_<training_size>_metrics.json
```

Training output:

- Checkpoint: `models/<model_id>.pt`
- Metrics report: `runs/training/<model_id>_metrics.json`
- Console progress: device selection, epochs, batches, and final metrics unless `--quiet` is used.
- Metrics include train/test loss, train/test MAE, continuous-parameter MAE, per-parameter MAE, waveform accuracy, and optional benchmark metrics.

New PyTorch models train waveform parameters with classification heads by default. Use `--waveform-mode scalar_regression` only for legacy comparison runs.

For pitch-conditioned timbre training, remove `freq` from the output target and feed exact synthetic pitch as an input channel:

```bash
python scripts/train_torch.py \
  --model-id vN_pytorch_cnn_pitchctx_<training_size> \
  --tensor-data data/generated/dN/features/mel_tensors.npz \
  --target-mode pitch_conditioned_timbre \
  --model-output models/vN_pytorch_cnn_pitchctx_<training_size>.pt \
  --metrics-output runs/training/vN_pytorch_cnn_pitchctx_<training_size>_metrics.json
```

Use `--loss-preset audibility` to weight waveform, detune, filter, and envelope parameters more strongly than low-impact normalized parameters:

```bash
python scripts/train_torch.py \
  --model-id vN_pytorch_cnn_pitchctx_weighted_<training_size> \
  --tensor-data data/generated/dN/features/mel_tensors.npz \
  --target-mode pitch_conditioned_timbre \
  --loss-preset audibility \
  --model-output models/vN_pytorch_cnn_pitchctx_weighted_<training_size>.pt \
  --metrics-output runs/training/vN_pytorch_cnn_pitchctx_weighted_<training_size>_metrics.json
```

For longer model-quality runs, prefer explicit optimizer controls and best-validation checkpoint selection:

```bash
python scripts/train_torch.py \
  --model-id vN_pytorch_cnn_pitchctx_weighted_<training_size> \
  --tensor-data data/generated/dN/features/mel_tensors.npz \
  --target-mode pitch_conditioned_timbre \
  --loss-preset audibility \
  --optimizer adamw \
  --weight-decay 0.01 \
  --scheduler step \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.5 \
  --early-stopping-patience 8 \
  --checkpoint-selection best_validation \
  --model-output models/vN_pytorch_cnn_pitchctx_weighted_<training_size>.pt \
  --metrics-output runs/training/vN_pytorch_cnn_pitchctx_weighted_<training_size>_metrics.json
```

## Evaluate PyTorch Models

Evaluate `v3_pytorch_cnn_500seeds` on `d2`:

```bash
python scripts/evaluate_dataset_torch.py \
  --metadata data/generated/d2/metadata.jsonl \
  --model models/v3_pytorch_cnn_500seeds.pt \
  --count 20 \
  --output runs/evaluation/v3_pytorch_cnn_500seeds_on_d2_eval.json
```

Evaluate `v4_pytorch_cnn_10kseeds` on `d3`:

```bash
python scripts/evaluate_dataset_torch.py \
  --metadata data/generated/d3/metadata.jsonl \
  --model models/v4_pytorch_cnn_10kseeds.pt \
  --count 200 \
  --start-index 8000 \
  --output runs/evaluation/v4_pytorch_cnn_10kseeds_on_d3_eval.json
```

Evaluation reports include weighted audio distance plus the component distances used to score predicted renders against target audio.

## Pitch And Length Strategy

For future model-quality work, `freq` should be treated as pitch context rather than a core timbre prediction target. Synthetic datasets already know the exact patch `freq`, so the model can use that value to interpret the mel spectrogram without spending output capacity predicting pitch.

For real one-note clips later, estimate pitch before prediction with a classical monophonic pitch estimator or allow manual pitch input.

Keep `length` visible to the model design for now because duration affects ADSR interpretation and whether a sound behaves like a pluck, key, or pad.

## Predict One Patch

Predict a patch JSON from one audio clip using a PyTorch checkpoint:

```bash
python scripts/predict_patch_torch.py \
  data/generated/d2/audio/patch_000000_seed_2000.wav \
  runs/pytorch_prediction/v3_pytorch_cnn_500seeds_patch_000000_seed_2000.json
```

Render and compare a PyTorch prediction against the target audio:

```bash
python scripts/evaluate_prediction_torch.py \
  data/generated/d2/audio/patch_000000_seed_2000.wav \
  --output-dir runs/pytorch_prediction/v3_pytorch_cnn_500seeds_patch_000000_seed_2000_eval
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
