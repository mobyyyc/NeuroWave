# NeuroWave PyTorch Runtime Decision

Decision: use PyTorch in the current project `.venv`.

## Current Runtime

- Local Python: `3.14.0`.
- Platform: macOS arm64.
- PyTorch now officially recommends Python 3.10 through 3.14 on macOS.
- Local PyTorch install: `2.12.0`.
- Local MPS availability check: unavailable, so use CPU locally unless this changes.
- The existing scikit-learn baseline works and remains the regression baseline.

## Compatibility Decision

Use the existing Python 3.14 project environment for PyTorch work.

Reason:

- The current project `.venv` uses Python 3.14.
- Current PyTorch install documentation recommends Python 3.10 through 3.14 on macOS.
- The PyTorch release compatibility matrix lists recent PyTorch releases with Python
  3.14 support.
- Keeping one project environment is simpler now that the active Python version is
  inside the supported range.

## Dependency Policy

- Install PyTorch into the existing `.venv` for Milestone G work.
- Keep PyTorch code isolated enough that the existing scikit-learn baseline remains usable.
- Add PyTorch dependency documentation before relying on it for normal setup.
- The local install command is:

```bash
python -m pip install torch
```

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
