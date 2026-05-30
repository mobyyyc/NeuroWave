# NeuroWave PyTorch Runtime Decision

Decision: do not add PyTorch to the current project `.venv` yet.

## Current Runtime

- Local Python: `3.14.0`.
- Platform: macOS arm64.
- `torch` is not currently installed.
- The existing scikit-learn baseline works and remains the regression baseline.

## Compatibility Decision

Use a separate Python 3.12 environment for PyTorch work unless official PyTorch macOS
wheel support for the active local Python version is confirmed at implementation time.

Reason:

- The current project `.venv` uses Python 3.14.
- PyTorch's macOS install documentation recommends Python 3.9 through 3.12.
- Adding PyTorch directly to the current `.venv` would risk destabilizing the working
  baseline and test environment.

## Dependency Policy

- Do not add `torch` to `requirements.txt` yet.
- Do not install PyTorch as part of ordinary setup yet.
- When Milestone G implementation begins, create an explicit PyTorch setup path, likely:

```bash
python3.12 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch
```

If Python 3.12 is unavailable locally, install or select a compatible Python runtime
before adding PyTorch code.

## Training Runtime

Initial PyTorch work should target CPU or Apple Silicon MPS if available.

The first PyTorch goal is correctness and comparison against the scikit-learn baseline,
not GPU optimization.

## Baseline Rule

Keep the scikit-learn MLP path until a PyTorch spectrogram model:

- trains successfully,
- predicts valid patches,
- evaluates across dataset clips,
- beats the scikit-learn baseline on weighted audio distance.

Only after that should the project decide whether to remove scikit-learn or keep it as
a lightweight sanity-check baseline.
