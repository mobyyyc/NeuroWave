# NeuroWave

NeuroWave is a Python synthesizer project being built toward machine-learning-driven sound recreation.

The current focus is a deterministic, parameterized synth engine that can render editable patch settings into audio. The repository and Python package still use the historical `MiniSynth` / `minisynth` names internally.

## Setup

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Render A Preset

Render the current `dark_saw` JSON preset:

```bash
python scripts/render_patch.py presets/dark_saw.json dark_saw.wav
```

## Generate A Small Dataset

Generate 10 paired random patch JSON and WAV files, plus metadata:

```bash
python scripts/random_patch.py --seed 1000 --count 10
```

Outputs are written under `data/generated/v1/`, which is ignored by git.

Generate a larger versioned dataset for scaled training:

```bash
python scripts/random_patch.py --dataset-version v2 --seed 2000 --count 500
```

## Train The MLP Baseline

Train the current scikit-learn MLP baseline on generated metadata:

```bash
python scripts/train_mlp.py --metadata data/generated/v1/metadata.jsonl
```

The command saves a checkpoint to `models/mlp_baseline.joblib` and prints train/test parameter MAE metrics as JSON. The `models/` directory is ignored by git.

Save a metrics report while training:

```bash
python scripts/train_mlp.py --metadata data/generated/v2/metadata.jsonl --metrics-output runs/training/v2_mlp_metrics.json
```

## PyTorch Runtime

PyTorch work is planned in a separate runtime path. See `PYTORCH_DECISION.md` before installing or adding PyTorch dependencies.

Export generated audio as channel-first mel-spectrogram tensors for future PyTorch training:

```bash
python scripts/export_mel_tensors.py --dataset-version v2
```

## Predict A Patch

Predict a patch JSON from one audio clip using the saved MLP checkpoint:

```bash
python scripts/predict_patch.py data/generated/v1/audio/patch_000000_seed_1000.wav runs/predicted_patch.json
```

Evaluate the prediction by rendering it and comparing it to the target:

```bash
python scripts/evaluate_prediction.py data/generated/v1/audio/patch_000000_seed_1000.wav
```

Evaluate a model across multiple dataset clips:

```bash
python scripts/evaluate_dataset.py --metadata data/generated/v2/metadata.jsonl --count 20 --output runs/evaluation/v2_mlp_eval.json
```

Compare two evaluation reports:

```bash
python scripts/compare_evaluation_reports.py runs/evaluation/v1_on_v2_eval.json runs/evaluation/v2_on_v2_eval.json --output runs/evaluation/v1_vs_v2_comparison.json
```

Optionally refine the ML prediction with a short local parameter search:

```bash
python scripts/evaluate_prediction.py data/generated/v1/audio/patch_000000_seed_1000.wav --refine-iterations 10
```

## Test

Run the unit tests:

```bash
python -m unittest discover -s tests
```

Run the smoke render:

```bash
python scripts/smoke_render.py
```
