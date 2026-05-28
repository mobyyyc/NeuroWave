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

## Test

Run the unit tests:

```bash
python -m unittest discover -s tests
```

Run the smoke render:

```bash
python scripts/smoke_render.py
```
