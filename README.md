# NeuroWave

NeuroWave is a Python synthesizer project being built toward machine-learning-driven sound recreation.

The current system renders known synth parameters into audio, generates synthetic datasets, trains inverse models from audio features, and evaluates predicted synth patches by rendering them back to audio.

The next product direction is a Windows-first desktop app: drag in a clean one-note audio
clip, crop the useful region, provide or confirm pitch, run the current PyTorch model,
and compare the rendered prediction against the target with waveform, spectrogram, WAV,
and JSON outputs.

Architecture note: the repository and Python package still use the historical `MiniSynth` / `minisynth` names internally. Do not rename folders, imports, or package paths unless that migration is planned separately.

## Project Map

- `minisynth/`: synth engine, schema, audio features, dataset helpers, and ML model code.
- `presets/`: human-readable synth presets.
- `scripts/`: commands for rendering, dataset generation, training, prediction, and evaluation.
- `data/generated/`: local generated datasets. Ignored by git.
- `models/`: local trained checkpoints. Ignored by git.
- `runs/`: local training, prediction, and evaluation reports. Ignored by git.
- `playground/`: local manual prediction experiments and app-style outputs. Ignored by git.
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
`v3.1_500ksamples`, `v3.2_oscmix`, `v3.3_main_detuned_mix`,
`v3.4_audible_loss`, and `v3.5_noise_detune_loss`. Keep dataset IDs and model IDs separate.

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
audibility-aware group-balanced loss, AdamW, step LR, early stopping, and best-validation
checkpoint saving. These are report fields, not model-name tokens.

Template for future models:

```bash
python scripts/train_torch.py \
  --model-id v3.5_noise_detune_loss \
  --tensor-data data/generated/dN/features \
  --epochs 50 \
  --batch-size 128 \
  --device cuda \
  --model-output models/v3.5_noise_detune_loss.pt \
  --metrics-output runs/training/v3.5_noise_detune_loss_metrics.json
```

Training output:

- Checkpoint: `models/<model_id>.pt`
- Metrics report: `runs/training/<model_id>_metrics.json`
- Console progress: device selection, epochs, batches, and final metrics unless `--quiet` is used.
- Metrics include train/test loss, train/test MAE, continuous-parameter MAE,
  per-parameter MAE, grouped MAE, waveform accuracy, and compact loss-history tails.

The current best proven model family is the v3 pitch-conditioned grouped-head setup. It keeps exact synthetic pitch as context, uses separate heads for duration, oscillator, filter, and ADSR controls, and trains with group-balanced loss so one parameter group cannot hide another.

`v3.1` scaled that setup to 500k synthetic examples. `v3.2` added oscillator-mix targets and proved that fixed oscillator slots create real target ambiguity, but the fair d8 evaluation showed worse median rendered distance because canonical slot swapping conflicts with `osc2_detune`.

`v3.3_main_detuned_mix` keeps pitch conditioning and total-level/balance learning, but defines oscillator roles by producer logic: `osc1` is the main/base-frequency oscillator supplied by the known pitch context, and `osc2` is the detuned oscillator whose relative pitch is learned as `detune_amount`. This preserves detune meaning while still avoiding independent raw `osc1_level`/`osc2_level` regression.

`v3.4_audible_loss` keeps the v3.3 target and architecture, but changes the training objective so waveform, balance, and detune mistakes matter more when the affected oscillator is audible and matter less when it is nearly silent.

The next planned model is `v3.5_noise_detune_loss`. It keeps v3.4's audibility-aware objective, but reduces detune loss when the detuned oscillator is noise because noise does not carry a meaningful pitched offset. It also boosts audible noise waveform classification so the model learns noise identity instead of chasing noise detune labels.

The v3 defaults are intentionally not exposed as routine CLI switches. If a future
experiment needs a different architecture or loss, change the code deliberately and
record that change as the model suffix.

## Evaluate PyTorch Models

Evaluate a current PyTorch checkpoint on a generated dataset:

```bash
python scripts/evaluate_dataset_torch.py \
  --metadata data/generated/dN/metadata.jsonl \
  --model models/v3.5_noise_detune_loss.pt \
  --count 1000 \
  --start-index 0 \
  --device cuda \
  --output runs/evaluation/v3.5_noise_detune_loss_on_dN_eval.json
```

Use a smaller count for quick checks:

```bash
python scripts/evaluate_dataset_torch.py \
  --metadata data/generated/dN/metadata.jsonl \
  --model models/v3.5_noise_detune_loss.pt \
  --count 200 \
  --output runs/evaluation/v3.5_noise_detune_loss_smoke_eval.json
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
  runs/pytorch_prediction/v3.5_noise_detune_loss_patch.json \
  --model models/v3.5_noise_detune_loss.pt \
  --freq 440
```

Render and compare a PyTorch prediction against the target audio:

```bash
python scripts/evaluate_prediction_torch.py \
  data/generated/dN/audio/patch_000000_seed_0000.wav \
  --model models/v3.5_noise_detune_loss.pt \
  --freq 440 \
  --output-dir runs/pytorch_prediction/v3.5_noise_detune_loss_patch_eval
```

## Compare Reports

Compare two evaluation reports:

```bash
python scripts/compare_evaluation_reports.py \
  runs/evaluation/v2_sklearn_mlp_500seeds_on_d2_eval.json \
  runs/evaluation/v3_pytorch_cnn_500seeds_on_d2_eval.json \
  --output runs/evaluation/v2_sklearn_mlp_500seeds_vs_v3_pytorch_cnn_500seeds_on_d2.json
```

## Product Prototype Direction

The first application target is Windows x64. The app should run local inference rather
than training: users drag in a WAV, crop a one-note region, enter/confirm frequency, run
prediction with a selected checkpoint, then listen to and export the rendered prediction.

Planned output files for each app run:

- cropped target WAV
- predicted patch JSON
- predicted rendered WAV
- target spectrogram artifact
- predicted spectrogram artifact
- summary JSON with model/checkpoint metadata

Start the local app backend:

```bash
python scripts/app_backend.py --host 127.0.0.1 --port 8765
```

Check that it is alive:

```bash
python -c "import requests; print(requests.get('http://127.0.0.1:8765/health').json())"
```

Prediction endpoint:

```text
POST http://127.0.0.1:8765/predict
Content-Type: application/json

{
  "audio_path": "playground/testpluck.wav",
  "model_path": "models/v3.5_noise_detune_loss.pt",
  "freq_hz": 440,
  "crop_start_seconds": 0.0,
  "crop_end_seconds": 0.4,
  "output_dir": "runs/app",
  "device": "cpu"
}
```

The website should come after the first usable desktop prototype and should publish the
product story, screenshots, A/B examples, model limitations, and Windows download or
waitlist flow.

Run the static frontend prototype:

```bash
python -m http.server 5173 --directory app
```

Then open:

```text
http://127.0.0.1:5173
```

After a prediction, the frontend loads the generated patch JSON, predicted WAV,
target spectrogram, and predicted spectrogram through the local backend artifact
endpoint. The endpoint only serves artifacts registered by successful predictions
in the current backend process.

The prototype also supports crop zoom, predicted JSON/WAV browser downloads, and
opening the current run folder through the local backend on Windows.

Run the Electron desktop shell in development:

```bash
npm install
npm run desktop
```

The Electron shell loads `app/index.html` in a desktop window and starts
`scripts/app_backend.py` automatically if nothing is already listening on
`127.0.0.1:8765`. Set `NEUROWAVE_PYTHON` if you need to use a Python executable
other than the project-local `.venv`.

The default app surface is producer-facing: drag audio, crop, confirm pitch,
predict, inspect parameters, A/B audio, and save JSON/WAV. Backend URL, model
path, raw audio path, output folder, and raw response JSON remain available only
as advanced/developer controls during the prototype phase.

Optional development settings can be copied from `desktop/settings.example.json`
to `desktop/settings.local.json`. The local file is ignored by git and can set
the Python executable, backend host/port, default model checkpoint, and default
output folder. Advanced app fields also persist in browser/Electron local
storage after editing.

Build the first Windows desktop development package:

```bash
npm run package:win
```

This creates an ignored portable `.exe` under `dist/`. It packages the Electron app
shell and NeuroWave Python source resources, but it is not yet the final consumer
installer: Python dependencies and the selected model checkpoint must still be
available on the machine. In a packaged build, optional `settings.local.json` can be
placed beside the `.exe` to override Python, backend port, default model path, and
output folder.

Packaged development builds search common repo/package locations for
`.venv/Scripts/python.exe` and `models/v3.5_noise_detune_loss.pt`, then pass absolute
model/output paths to the frontend. Backend startup logs are written to
`neurowave-backend.log` beside the packaged executable by default.

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
