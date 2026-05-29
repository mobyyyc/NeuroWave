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

## Train The MLP Baseline

Train the current scikit-learn MLP baseline on generated metadata:

```bash
python scripts/train_mlp.py --metadata data/generated/v1/metadata.jsonl
```

The command saves a checkpoint to `models/mlp_baseline.joblib` and prints train/test parameter MAE metrics as JSON. The `models/` directory is ignored by git.

## Test

Run the unit tests:

```bash
python -m unittest discover -s tests
```

Run the smoke render:

```bash
python scripts/smoke_render.py
```
