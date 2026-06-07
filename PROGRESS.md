# NeuroWave Progress Tracker

Project brand: NeuroWave.

Architecture note: the local folder, GitHub repo, and Python package currently remain `MiniSynth` / `minisynth`. Do not rename files, folders, imports, or package paths unless explicitly requested.

## How To Use This Tracker

This file is the day-to-day execution layer on top of `PLAN.md`.

For the full new-session operating procedure, read `WORKFLOW.md` first.

Use `PLAN.md` for the long-term architecture and reasoning. Use this file for concrete work items, status, and daily progress.

Rules for future work:

- Before starting work, read `PLAN.md` and this file.
- Pick a small unchecked task that can reasonably be completed in one focused session.
- Prefer tasks from the current active phase before jumping ahead.
- After completing a task, change `[ ]` to `[x]`.
- Add a short note under "Progress Log" with the date, what changed, and the commit hash.
- Commit after each completed task or coherent group of tiny tasks.
- Push immediately after each commit.
- Keep generated files such as `.venv/`, `*.wav`, `data/generated/`, `runs/`, and `models/` out of git unless there is a deliberate reason.
- If the project direction changes, update `PLAN.md` first, then update this tracker.

Status meanings:

- `[ ]` Not started.
- `[x]` Done and committed.
- `[~]` In progress. Avoid leaving this state at the end of a session.
- `[!]` Blocked. Add a note explaining the blocker.

Current active phase: Milestone I - Product Prototype - Windows Desktop App.

## Completed Baseline

- [x] Clone full `mobyyyc/MiniSynth` repo into `Desktop/Coding/MiniSynth`.
- [x] Create project-local `.venv` inside `MiniSynth`.
- [x] Install dependencies into `MiniSynth/.venv`.
- [x] Run current `synth.py` successfully and generate `dark_saw.wav`.
- [x] Add long-term roadmap in `PLAN.md`.
- [x] Update `.gitignore` for local environments, caches, generated audio, datasets, runs, and model outputs.
- [x] Add initial package scaffold in `minisynth/`, plus `presets/`, `scripts/`, and `tests/`.

## Milestone A: Clean Synth Engine

Goal: turn the current script into a modular, deterministic, parameterized synth engine without changing the sound accidentally.

- [x] Add a tiny smoke test command or script that renders the current `dark_saw` path.
- [x] Move `SR = 44100` into a central constants location.
- [x] Move `adsr()` from `synth.py` into `minisynth/envelopes.py`.
- [x] Add a focused unit test for `adsr()` length and value range.
- [x] Move `oscillator()` into `minisynth/oscillators.py`.
- [x] Add triangle waveform support.
- [x] Add wave-mix rendering for sine, triangle, saw, square, and noise.
- [x] Add tests for wave-mix normalization.
- [x] Move `lowpass_filter()` into `minisynth/filters.py`.
- [x] Add tests that low-pass filtering preserves array length and finite values.
- [x] Move `render_patch()` into `minisynth/engine.py`.
- [x] Make `render_patch()` return audio only and never save files or show plots.
- [x] Add tests for deterministic rendering from the same patch.
- [x] Create initial `SynthConfig` data model in `minisynth/schema.py`.
- [x] Define parameter metadata type with name, kind, min, max, default, scale, group, and `ml_enabled`.
- [x] Add linear normalization and denormalization helpers.
- [x] Add logarithmic normalization and denormalization helpers.
- [x] Add tests for normalization round trips.
- [x] Convert `PATCHES["dark_saw"]` into `presets/dark_saw.json`.
- [x] Add `minisynth/io.py` helpers to load and save JSON patches.
- [x] Add `scripts/render_patch.py` that renders a preset JSON to a WAV path.
- [x] Update `README.md` with setup and render commands.
- [x] Remove old `synth.py` after scripts replaced it.
- [x] Run the new render script and verify it recreates `dark_saw.wav`.
- [x] Commit Milestone A completion.

## Milestone B: Dataset Tools

Goal: generate labeled synthetic data from known parameters.

- [x] Create `scripts/random_patch.py` with seeded random patch generation.
- [x] Add constraints so random patches are not silent.
- [x] Add constraints so generated audio avoids clipping.
- [x] Save generated patch JSON files under `data/generated/d1/params/`.
- [x] Save generated WAV files under `data/generated/d1/audio/`.
- [x] Write `metadata.jsonl` for generated clips.
- [x] Add a small sample generation command to `README.md`.
- [x] Generate a tiny local dataset of 10 patches for manual inspection.
- [x] Add tests for reproducible random patch generation from a seed.
- [x] Commit Milestone B completion.

## Milestone C: Audio Features And Similarity

Goal: compare target audio and rendered audio in a useful, phase-tolerant way.

- [x] Add mono conversion helper.
- [x] Add resampling helper.
- [x] Add loudness normalization helper.
- [x] Add mel spectrogram extraction in `minisynth/features.py`.
- [x] Add RMS envelope extraction.
- [x] Add multi-resolution STFT magnitude extraction.
- [x] Add spectral centroid extraction.
- [x] Add `scripts/compare_audio.py`.
- [x] Define first weighted similarity score.
- [x] Test that identical audio scores better than different audio.
- [x] Commit Milestone C completion.

## Milestone D: Search-Based Matching

Goal: match target audio by optimizing synth parameters before training neural networks.

- [x] Add parameter vector to `SynthConfig` conversion.
- [x] Add vector to `SynthConfig` reconstruction.
- [x] Add random search over parameter vectors.
- [x] Save best candidate patch and WAV into `runs/`.
- [x] Add progress output for search runs.
- [x] Test matching against a NeuroWave-generated target.
- [x] Add a simple comparison report file in each run directory.
- [x] Commit Milestone D completion.

## Milestone E: First ML Model

Goal: predict synth parameters from audio features on synthetic data.

- [x] Choose initial ML framework.
- [x] Create dataset loader for generated metadata.
- [x] Create first scikit-learn MLP regression baseline.
- [x] Train on a small synthetic dataset.
- [x] Save model checkpoints under `models/`.
- [x] Add prediction script for one audio clip.
- [x] Render predicted patch and compare to target.
- [x] Add optional optimizer refinement after prediction.
- [x] Commit Milestone E completion.

## Milestone F: Scaled Synthetic Training

Goal: make the first ML model learn from larger synthetic datasets before moving to real audio.

- [x] Add versioned dataset output support for `data/generated/d2/`.
- [x] Fix random patch envelope constraints so generated targets stay inside schema ranges.
- [x] Generate a local ignored `d2` dataset with 500 examples.
- [x] Train the MLP baseline on the `d2` dataset.
- [x] Save a metrics report for the `d2` training run.
- [x] Add evaluation across multiple synthetic dataset clips.
- [x] Compare the `v1_sklearn_mlp_10seeds` model against the `v2_sklearn_mlp_500seeds` model.
- [x] Decide whether to keep scikit-learn MLP or move next to PyTorch.
- [x] Commit Milestone F completion.

## Milestone G: PyTorch Spectrogram Model

Goal: move beyond the compact scikit-learn baseline toward a richer inverse model that learns from spectrogram structure.

- [x] Make an explicit PyTorch dependency and runtime decision.
- [x] Create mel-spectrogram dataset tensor export from generated metadata.
- [x] Build first PyTorch inverse model for normalized `SynthConfig` vectors.
- [x] Train on `d2` or a larger synthetic dataset.
- [x] Save PyTorch checkpoints and metrics.
- [x] Add prediction script for one clip using the PyTorch model.
- [x] Render PyTorch predicted patch and compare to target.
- [x] Evaluate PyTorch model across synthetic dataset clips.
- [x] Compare PyTorch directly against the scikit-learn baseline.
- [x] Decide whether scikit-learn stays as a lightweight baseline or can be removed.
- [x] Generate `d3` with 10,000 seeds and train `v4_pytorch_cnn_10kseeds`.
- [x] Commit Milestone G completion.

## Milestone H: Model Capability And Target Quality

Goal: make the PyTorch inverse model strong enough that the model itself is not the bottleneck when trained on a good synthetic dataset.

Evidence from local reports:

- `v5_pytorch_cnn_10kseeds`: `test_loss = 0.0461`, `test_mae = 0.1616`.
- `v8_pytorch_cnn_50kseeds`: `test_loss = 0.0375`, `test_mae = 0.1339`.
- `v9_pytorch_cnn_200kseeds`: `test_loss = 0.0356`, `test_mae = 0.1284`.
- `v10_pytorch_cnn_pitchctx_weighted_medium_tfpool_50kseeds`: best rendered-audio result in the first capability set, but weaker parameter metrics.
- `v2.0_pytorch_cnn_pitchctx_flat_medium_tfpool_50kseeds`: better parameter MAE and waveform accuracy than `v10`, but worse rendered-audio distance.
- `v2.1_pytorch_cnn_pitchctx_hybrid_medium_tfpool_50kseeds`: best parameter MAE and waveform accuracy in the capability set, but worse rendered-audio distance because filter/cutoff errors grew.
- Train/test metrics remain close, which suggests undercapacity, target representation limits, or loss design limits more than overfitting.
- Prediction spread diagnostics show strong regression toward average values for oscillator levels, resonance, ADSR timing, and release. The next target is to push toward `test_mae <= 0.05` on a fixed synthetic holdout while fixing this mean-collapse behavior and improving rendered-audio distance.

- [x] Add per-parameter MAE reporting to PyTorch training metrics.
- [x] Add waveform prediction metrics separate from continuous-parameter MAE.
- [x] Add a fixed benchmark split or benchmark dataset that is not used for training or tuning.
- [x] Add model comparison reporting across parameter metrics and rendered-audio metrics.
- [x] Replace scalar waveform enum regression with classification heads or continuous wave-mix targets.
- [x] Decide pitch/length handling strategy for the next model-design iteration.
- [x] Add target groups for pitch-conditioned timbre metrics, ADSR metrics, oscillator metrics, and filter metrics.
- [x] Remove `freq` from the core timbre prediction target while feeding exact synthetic `freq` as model conditioning.
- [x] Keep `length` visible to the model design while evaluating how it interacts with ADSR and pluck/pad behavior.
- [x] Add parameter-weighted loss support.
- [x] Add optimizer and training controls: AdamW, weight decay, scheduler, early stopping, and best-validation checkpoint saving.
- [x] Build a larger scalable PyTorch model family with deeper/wider CNN or residual blocks.
- [x] Preserve more time-frequency structure before pooling in the model architecture.
- [x] Add worst-clip diagnostics that compare target and predicted synth parameters for rendered-audio failures.
- [x] Add hybrid loss preset for the next v2.1 pitch-conditioned model.
- [x] Train and evaluate first capability models against `v9_pytorch_cnn_200kseeds`.
- [x] Document whether remaining error is model capacity, target ambiguity, dataset quality, or synth parameter non-uniqueness.
- [x] Add prediction distribution diagnostics for target-vs-predicted mean and standard deviation by parameter.
- [x] Fix non-finite rendered prediction handling so invalid predicted patches are captured and clamped or reported safely.
- [x] Implement the `v3.0` model-design step: pitch-conditioned, group-balanced, multi-head continuous prediction.
- [x] Train and evaluate `v3.0` against `v10`, `v2.0`, `v2.1`, and `v9`.
- [x] Scale the v3 setup to `v3.1` on 500k synthetic examples.
- [x] Implement `v3.2` oscillator-mix target improvements so oscillator levels are learned
  by canonical waveform/level contribution rather than arbitrary oscillator slot identity.
- [x] Implement the `v3.3` main/detuned oscillator target setup before training.
- [x] Implement the `v3.4` audibility-aware loss setup before training.
- [x] Implement the `v3.5` noise-aware detune loss setup before training.

Current v3 findings:

- `v3.0` confirmed that pitch conditioning, grouped heads, and group-balanced loss are
  the correct model family: parameter metrics and rendered-audio distance improved
  together.
- `v3.1` confirmed the setup scales with larger data: test MAE reached about `0.0715`,
  waveform accuracy reached about `0.882`, and d8 mean weighted distance reached about
  `32.2`.
- The active bottleneck is now oscillator mix representation, especially `osc1_level`,
  `osc2_level`, and `osc2_detune`.
- Because the two oscillator slots are partly exchangeable, the next improvement should
  make oscillator targets slot-invariant or canonical. A quiet saw in oscillator 1 plus
  a loud sine in oscillator 2 should not be treated as a fundamentally different target
  from the same saw/sine level contributions swapped between slots.

Completed v3.2 setup before training:

- Added oscillator-mix diagnostics before training: total level, balance, per-wave level
  contribution, slot-swapped/best-assignment error, and worst clips grouped by
  oscillator-mix failure.
- Added the `oscillator_mix` target mode used by the simplified future-facing training
  CLI. It uses pitch context, canonicalizes oscillator slot ordering, replaces raw
  `osc1_level`/`osc2_level` targets with `osc_total_level` and `osc_balance`, and
  reconstructs ordinary renderable levels during prediction.

v3.2 training/evaluation finding:

- `v3.2_oscmix` improved parameter metrics and waveform accuracy, but fair d8 evaluation
  showed worse median rendered distance versus v3.1 while reducing worst-case failures.
  The likely bottleneck is that canonical oscillator swapping conflicts with the synth's
  `osc2_detune` convention.

Completed v3.3 setup before training:

- Added the `main_detuned_mix` target mode. It keeps pitch context, treats `osc1` as
  the main/base-frequency oscillator, treats `osc2` as the detuned oscillator, predicts
  `osc_total_level`, `detuned_balance`, and `detune_amount`, then reconstructs normal
  render parameters after prediction.
- Added main/detuned diagnostics for evaluation reports: main wave error, detuned wave
  error, total-level error, detuned-balance error, and normalized detune error.
- Updated the simplified training CLI defaults to `v3.3_main_detuned_mix`.

v3.3 training/evaluation finding:

- `v3.3_main_detuned_mix` became the current best rendered-audio setup on the fixed d8
  1000-clip evaluation, improving mean and median weighted distance versus v3.1 and v3.2.
  It still has rare severe outliers from audible oscillator mistakes, especially wrong
  wave/detune on loud detuned oscillators and total-level overshoot on quiet targets.

Completed v3.4 setup before training:

- Added the `audible` loss preset. It keeps the group-balanced structure, but weights
  main-wave loss by main oscillator audibility, detuned-wave and detune loss by detuned
  oscillator audibility, detuned-balance loss by total oscillator audibility, and
  total-level overshoot more strongly for quiet targets.
- Updated the simplified training CLI defaults to `v3.4_audible_loss`.

v3.4 training/evaluation finding:

- `v3.4_audible_loss` became the best rendered-audio model so far on the fixed d8
  1000-clip evaluation. Remaining worst clips are still concentrated around audible
  noise waveform mistakes and large detune errors on noise oscillators, where detune is
  not a meaningful pitched offset.

Completed v3.5 setup before training:

- Added the `noise_detune` loss preset. It keeps the v3.4 audibility-aware objective,
  suppresses `detune_amount` loss when the detuned target wave is noise, and boosts
  audible noise waveform classification.
- Updated the simplified training CLI defaults to `v3.5_noise_detune_loss`.

Next recommended task:

- Train `v3.5_noise_detune_loss` on the current 500k tensor dataset and compare it against
  `v3.4_audible_loss` on the fixed d8 1000-clip evaluation.
- [ ] Decide whether the current waveform enum target must become continuous wave-mix before aiming for `test_mae <= 0.05`.
- [ ] Commit Milestone H completion.

## Milestone I: Product Prototype - Windows Desktop App

Goal: make NeuroWave usable by a producer without the terminal. The first platform is
Windows x64 because the current model workflow, CUDA setup, and local development
environment are already Windows-first.

Product decision:

- [x] Choose first desktop platform: Windows.
- [x] Keep training outside the first app; the app runs inference and rendering only.
- [x] Use the current best checkpoint family as the app model source.
- [x] Keep generated app outputs out of git.
- [x] Decide desktop shell after backend slice: Electron.

Backend/app inference foundation:

- [x] Create `minisynth/app_inference.py`.
- [x] Move reusable logic from `scripts/playground_predict_wav.py` into the app inference module.
- [x] Add an app request dataclass or structured dict for `audio_path`, crop start/end, `freq_hz`, `model_path`, `output_dir`, and run ID.
- [x] Add an app response dataclass or structured dict for output paths, model metadata, warnings, and errors.
- [x] Add robust WAV loading for mono/stereo audio.
- [x] Add crop validation for start/end seconds.
- [x] Add crop extraction and cropped target WAV export.
- [x] Add frequency validation and note that pitch context is required.
- [x] Add prediction using the selected PyTorch checkpoint.
- [x] Add predicted patch JSON export.
- [x] Add predicted audio render/export.
- [x] Add target spectrogram artifact export.
- [x] Add predicted spectrogram artifact export.
- [x] Add summary JSON export.
- [x] Add tests for crop validation.
- [x] Add tests for stereo-to-mono handling.
- [x] Add tests for deterministic app output paths.
- [x] Add tests for app response shape.
- [x] Add tests for invalid model/audio/frequency errors.

Local backend service:

- [x] Choose the first local inference API implementation. Decision: use a standard-library
  HTTP JSON server now because FastAPI/Uvicorn are not current project dependencies.
- [x] Add `scripts/app_backend.py` or equivalent local server entry point.
- [x] Add `/health`.
- [x] Add `/predict`.
- [x] Add request validation.
- [x] Add JSON response serialization.
- [x] Add backend error handling for invalid audio, invalid crop, missing model, and inference failure.
- [x] Add backend smoke test using a tiny fixture or generated clip.
- [x] Document local backend startup command.

Frontend prototype:

- [x] Choose frontend stack. Decision: start with dependency-free static HTML/CSS/JS.
- [x] Create app directory without disrupting existing Python package names.
- [x] Add app shell layout.
- [x] Add drag/drop WAV import.
- [x] Add waveform display.
- [x] Add crop start/end handles.
- [x] Add crop zoom.
- [x] Add crop playback.
- [x] Cap crop selection to the current model input window.
- [x] Add waveform playhead during crop playback.
- [x] Add frequency input in Hz.
- [x] Add note-name helper such as A4 -> 440 Hz.
- [x] Add model path/default model selector.
- [x] Add Predict button.
- [x] Add loading/progress state.
- [x] Connect Predict action to backend.
- [x] Display predicted patch JSON.
- [x] Display rendered predicted WAV player.
- [x] Display target spectrogram.
- [x] Display predicted spectrogram.
- [x] Add original/predicted comparison playback controls.
- [x] Add export predicted JSON button.
- [x] Add export predicted WAV button.
- [x] Add open run folder button.
- [x] Add error UI for unsupported audio, missing frequency, invalid crop, backend offline, and model failure.

Scaling note: the current static app fetches prediction artifacts through a local
backend `/artifact` endpoint that only serves files registered by successful
predictions in the current backend process. When the desktop wrapper is chosen,
replace or wrap this with a stronger app-shell file handoff so reloads, recent
runs, and packaged builds can reopen older run folders without weakening local
file access controls.

Foundation note: crop zoom, browser downloads, and `POST /open-folder` are now
implemented in the static prototype. The open-folder action is also allowlisted
to run folders registered by successful predictions in the current backend
process. Desktop packaging still needs a stronger native file handoff.

User-mode note: the default app surface is now producer-facing. Backend URL,
model checkpoint path, raw audio path, output directory, and raw response JSON
are treated as advanced/developer controls instead of the primary workflow.

Desktop packaging:

- [x] Choose Windows desktop wrapper: Electron.
- [x] Launch local Python backend from desktop app in development mode.
- [x] Add app setting for Python/backend path during development.
- [x] Add app setting for default model checkpoint.
- [x] Add Windows development package command and builder configuration.
- [x] Package a first Windows development build with `npm run package:win`.
- [x] Improve packaged development backend startup by auto-detecting nearby `.venv`,
  using absolute default model/output paths, and writing a backend startup log beside
  the packaged executable.
- [x] Fix packaged prediction setup by preferring desktop-supplied absolute defaults
  over stale stored frontend settings, resolving Electron dropped-file paths through
  the preload bridge, and logging prediction tracebacks to the backend log.
- [x] Replace fragile packaged audio path usage with an Electron import handoff that
  copies selected audio into an ignored app-controlled input folder before prediction.
- [x] Make backend health startup lightweight by lazy-loading the inference stack on
  `/predict`, and log Python startup success, timeout, spawn errors, and exits.
- [x] Surface desktop backend startup errors and log path in the app UI when backend
  health checks fail.
- [x] Auto-refresh backend readiness at app launch so slow backend startup does not leave
  the UI stuck in an early error state.
- [x] Verify the rebuilt unpacked and portable desktop apps start the backend and pass
  `/health` while running.
- [x] Store packaged app inputs, prediction runs, and backend logs under
  `%LOCALAPPDATA%\NeuroWave\` by default.
- [x] Verify the rebuilt unpacked Windows app starts the packaged backend, uses the bundled
  model checkpoint, predicts from a WAV, renders output audio, and writes a complete run
  under `%LOCALAPPDATA%\NeuroWave\Runs\`.
- [x] Add automated packaged app smoke commands for backend/runtime readiness and optional
  prediction artifact verification.
- [x] Manually verify the packaged UI drag/drop, visual crop, preview, predict, export JSON,
  export WAV, and open-folder flow.
- [x] Document Windows development package notes.
- [x] Add a prepared Python runtime packaging hook under ignored `runtime/python/`.
- [x] Add prepared Python runtime validation command.
- [x] Bundle or provision a stable Python runtime for non-developer machines.
- [ ] Test the packaged app on a clean Windows machine without the repo, project `.venv`,
  or developer Python installation.
- [x] Bundle the selected production model checkpoint when the local ignored `.pt` file
  exists at package time.
- [x] Rebuild the standalone portable Windows `.exe` after first-release UX updates.
- [x] Document Windows release checklist and current install/run notes.

Product UX polish:

- [x] Hide developer/runtime fields behind Advanced by default.
- [x] Add first-release readiness states for backend, model, audio, and crop validity.
- [x] Add recent input files.
- [x] Add recent prediction runs.
- [x] Add app output folder setting.
- [x] Add model/version display in prediction results.
- [x] Add CPU/CUDA runtime indicator.
- [x] Add warning when frequency is missing or likely wrong.
- [x] Add clear copy explaining best input: clean one-note clips.
- [x] Add limitations panel.
- [x] Polish the first-release app shell with clean product styling, clear states, and
  interactive hover/focus feedback.
- [ ] Add app screenshots for website use.

Acceptance checklist:

- [x] A user can drag a WAV into the app.
- [x] A user can crop the one-note region visually.
- [x] A user can preview the crop.
- [x] A user can enter frequency and run prediction.
- [x] The app produces predicted JSON and predicted WAV.
- [x] The app produces target and predicted spectrogram artifacts.
- [x] The app displays target and predicted spectrograms in the packaged UI.
- [x] The app can compare-play original crop and prediction.
- [x] The app saves a complete local run folder.
- [x] The app handles invalid inputs without crashing.
- [ ] Commit Milestone I completion.

## Milestone J: Product Website

Goal: publish NeuroWave as a product with a clear landing page, examples, and a Windows
download or waitlist flow.

Website planning:

- [ ] Choose website stack: Next.js, static HTML, or another lightweight framework.
- [ ] Choose hosting target.
- [ ] Define product positioning.
- [ ] Define first call to action: download, waitlist, or contact.
- [ ] Gather app screenshots.
- [ ] Gather audio comparison examples.
- [ ] Gather spectrogram comparison images.

Website build:

- [ ] Create website app or docs site directory.
- [ ] Add Home page.
- [ ] Add How It Works page.
- [ ] Add Examples page.
- [ ] Add Download page.
- [ ] Add Changelog page.
- [ ] Add responsive layout.
- [ ] Add audio players for examples.
- [ ] Add product screenshots.
- [ ] Add model/version labels for examples.
- [ ] Add known limitations copy.
- [ ] Add Windows requirements copy.
- [ ] Add privacy note explaining audio is processed locally in the desktop app.
- [ ] Add contact or waitlist form/link.
- [ ] Add basic SEO metadata.
- [ ] Add website build/test command.
- [ ] Deploy first website version.

Website acceptance checklist:

- [ ] First viewport clearly says what NeuroWave does.
- [ ] Website shows at least one audio comparison example.
- [ ] Website shows the drag/crop/predict app flow.
- [ ] Website explains Windows-first availability.
- [ ] Website provides a download or waitlist/contact path.
- [ ] Commit Milestone J completion.

## Milestone K: Product Hardening

Goal: make NeuroWave reliable enough for repeated use outside the developer environment.

- [x] Add local app logs.
- [ ] Add crash/error report files under app run directory.
- [ ] Add backend request/response logging without storing unnecessary user audio copies.
- [ ] Add app settings persistence.
- [ ] Add model/checkpoint selection UI.
- [ ] Add versioned app releases.
- [ ] Add installer packaging.
- [ ] Add Windows code-signing investigation.
- [x] Add packaged app smoke test.
- [x] Add release checklist.
- [ ] Add update/changelog workflow.
- [ ] Add `neurowave compare`.
- [ ] Add `neurowave match`.
- [ ] Add `neurowave predict`.
- [ ] Consider a simple local web UI after CLI is stable.
- [ ] Commit Milestone J completion.

## Progress Log

### 2026-06-06

- Added a backend smoke-flow test that creates a tiny WAV fixture, posts it through
  `/predict`, verifies `/health` and `/runtime`, fetches registered JSON/WAV artifacts
  through `/artifact`, and checks registered run-folder opening. Verified
  `.\.venv\Scripts\python.exe -m unittest tests.test_app_backend` and
  `.\.venv\Scripts\python.exe -m unittest discover -s tests`.
  Commit: `Add backend smoke flow test`
- Recorded manual packaged UI verification after the drag/drop, crop, preview, predict,
  spectrogram display, original/predicted compare-play, JSON/WAV export, and open-folder
  flow passed in the packaged app.
  Commit: `Record packaged UI verification`
- Added a Windows runtime validator and npm commands for checking both the development
  `.venv` dependency shape and the ignored `runtime/python/` candidate before packaging.
  Documented the first-release runtime preparation flow and kept clean-machine runtime
  readiness open until a real bundled runtime is produced and tested.
  Commit: `Add Windows runtime validation command`
- Added a repeatable `npm run runtime:prepare` flow that stages the active Python 3.14
  runtime and working dependency set into ignored `runtime/python/`, filters stale venv
  markers out of packaged `resources/python-runtime/`, rebuilt `dist/win-unpacked`, and
  verified packaged backend health plus prediction smoke against the bundled runtime.
  Clean-machine verification remains the next external release gate.
  Commit: `Prepare bundled Windows runtime`

### 2026-06-05

- Finalized the first-release desktop app UX foundation: readiness states for backend,
  model, audio, and crop validity; safer Predict-button gating; backend startup polling;
  model checkpoint bundling in packaged resources; model label in prediction results; and
  polished product styling.
  Commit: `Finalize desktop app UX`
- Verified JavaScript parse checks, package JSON parsing, CSS variable definitions, the
  full Python unit suite, and a direct app-inference smoke run against
  `playground/testpluck.wav` that produced target crop, predicted patch, predicted WAV,
  target spectrogram, predicted spectrogram, and summary artifacts under ignored
  `runs/app_smoke/`.
  Commit: `Finalize desktop app UX`
- Packaging and in-app browser verification were not completed in this pass because
  Electron Builder was blocked by the local approval/usage gate and the browser plugin
  blocked both localhost and file URLs by policy. The packaged end-to-end WAV import,
  crop, predict, render, and export checklist item remains open.
  Commit: `Finalize desktop app UX`
- Rebuilt the unpacked Windows app with `npm run package:win:dir`, confirmed the bundled
  checkpoint exists under `dist/win-unpacked/resources/models/`, smoke-launched
  `dist/win-unpacked/NeuroWave.exe`, verified `/health`, and posted a packaged-backend
  prediction request using `playground/testpluck.wav`. The run produced `target_crop.wav`,
  `predicted_patch.json`, `predicted.wav`, both spectrogram JSON files, and `summary.json`
  under `%LOCALAPPDATA%\NeuroWave\Runs\packaged_backend_testpluck_final\`.
  Commit: `Verify packaged app backend`
- Added the remaining first-release app UX pieces: runtime CPU/CUDA indicator, separate
  `/runtime` backend endpoint, pitch mismatch warning, recent files, recent runs, clearer
  original/predicted compare labels, and a compact limitations panel. Rebuilt
  `dist/win-unpacked/NeuroWave.exe`, rebuilt the standalone `dist/NeuroWave 0.1.0.exe`,
  verified backend `/health` and `/runtime`, and smoke-launched both desktop builds.
  Commit: `Add first release app UX`
- Added an ignored `runtime/python/` convention for prepared Python/Torch release
  runtimes, taught the desktop wrapper to prefer a bundled `resources/python-runtime`
  interpreter before falling back to developer `.venv` paths, and documented the current
  Windows release checklist in `docs/WINDOWS_RELEASE.md`. Verified `npm run package:win:dir`
  still builds and the app still smoke-starts with `.venv` fallback when no prepared
  runtime is supplied.
  Commit: `Add Windows runtime packaging hook`
- Added `npm run package:smoke` and `npm run package:smoke:predict` so the unpacked
  packaged app can be checked for backend health, runtime readiness, and end-to-end
  prediction artifacts. Verified both commands against the current `dist/win-unpacked`
  app; the prediction smoke wrote outputs under
  `%LOCALAPPDATA%\NeuroWave\Runs\desktop_package_smoke\`.
  Commit: `Add packaged app smoke test`

### 2026-05-27

- Created long-term roadmap in `PLAN.md`.
  Commit: `d26697a Add MiniSynth roadmap`
- Expanded `.gitignore` so local development and generated outputs stay out of git.
  Commit: `1bec0d4 Ignore local development artifacts`
- Added initial `minisynth/` package scaffold and placeholder project directories.
  Commit: `f7b1e8b Add initial MiniSynth package scaffold`
- Added `scripts/smoke_render.py` to verify the current `dark_saw` render path before refactoring.
  Commit: `Add render smoke test`
- Moved the sample-rate constant into `minisynth/constants.py` while keeping current render behavior.
  Commit: `Move sample rate into constants`
- Moved the ADSR envelope generator into `minisynth/envelopes.py`.
  Commit: `Move ADSR into envelopes module`
- Added a focused standard-library unit test for ADSR envelope length and value range.
  Commit: `Add ADSR envelope unit test`
- Moved the existing sine, saw, and square oscillator generator into `minisynth/oscillators.py`.
  Commit: `Move oscillator into oscillators module`
- Added triangle waveform support to the oscillator generator.
  Commit: `Add triangle oscillator waveform`
- Added normalized wave-mix rendering for sine, triangle, saw, square, and deterministic noise.
  Commit: `Add oscillator wave-mix rendering`
- Added focused unit tests for wave-mix normalization and mixed waveform output.
  Commit: `Add wave-mix normalization tests`
- Moved the low-pass filter implementation into `minisynth/filters.py`.
  Commit: `Move low-pass filter into filters module`
- Added focused unit tests for low-pass filter length preservation and finite output.
  Commit: `Add low-pass filter unit tests`
- Moved the patch renderer into `minisynth/engine.py`.
  Commit: `Move render patch into engine module`
- Added an engine test proving `render_patch()` returns audio without writing a WAV file.
  Commit: `Add render patch purity test`
- Added an engine test proving identical patch parameters render identical audio.
  Commit: `Add deterministic render test`
- Added initial `SynthConfig` dataclass and tests for defaults plus render kwargs conversion.
  Commit: `Add initial SynthConfig model`
- Added parameter metadata definitions for current `SynthConfig` fields.
  Commit: `Add parameter metadata schema`
- Renamed project-facing brand references from MiniSynth to NeuroWave while preserving package and repo architecture.
  Commit: `Rename project branding to NeuroWave`
- Added linear normalization and denormalization helpers with focused schema tests.
  Commit: `Add linear normalization helpers`
- Added logarithmic normalization and denormalization helpers with focused schema tests.
  Commit: `Add logarithmic normalization helpers`
- Added metadata-driven tests for numeric parameter normalization round trips.
  Commit: `Add normalization round-trip tests`
- Added `presets/dark_saw.json` as the first data-file version of the existing Python preset.
  Commit: `Add dark saw JSON preset`
- Added JSON patch load/save helpers and tests.
  Commit: `Add JSON patch IO helpers`
- Added a preset render script that loads JSON patch files and writes WAV output.
  Commit: `Add JSON preset render script`
- Updated the README with setup, render, test, and smoke render commands.
  Commit: `Update README with render workflow`
- Kept `synth.py` as a compatibility wrapper around the JSON preset render path.
  Commit: `Make synth.py a compatibility wrapper`
- Removed the historical `synth.py` wrapper after the smoke render moved to the JSON preset path.
  Commit: `Remove legacy synth.py wrapper`
- Completed Milestone A and moved the active phase to Milestone B dataset tools.
  Commit: `Complete Milestone A`
- Added seeded random patch generation for creating schema-valid parameter JSON.
  Commit: `Add seeded random patch generator`
- Added random patch constraints so generated patches render non-silent finite audio.
  Commit: `Prevent silent random patches`
- Added a reusable audio clipping constraint for generated dataset audio.
  Commit: `Add generated audio clipping constraint`
- Added dataset helpers and CLI support for saving generated patch JSON files under `data/generated/d1/params/`.
  Commit: `Save generated patch params`
- Added dataset helpers and CLI support for saving generated WAV files under `data/generated/d1/audio/`.
  Commit: `Save generated dataset audio`
- Added `metadata.jsonl` generation so each dataset row links seed, patch JSON, and rendered WAV paths.
  Commit: `Write generated dataset metadata`
- Documented the small dataset generation command in the README.
  Commit: `Document dataset generation command`
- Generated the local ignored 10-example dataset under `data/generated/d1/` for manual inspection.
  Commit: `Generate local sample dataset`
- Added tests proving random patch and dataset generation are reproducible for the same seed.
  Commit: `Add random generation reproducibility tests`
- Completed Milestone B and moved the active phase to Milestone C audio features and similarity.
  Commit: `Complete Milestone B`
- Added a mono conversion helper for one-dimensional and channel-last audio inputs.
  Commit: `Add mono conversion helper`
- Added a resampling helper for converting audio to the target sample rate.
  Commit: `Add audio resampling helper`
- Added RMS measurement and loudness normalization helpers for audio preprocessing.
  Commit: `Add loudness normalization helper`
- Added log-mel spectrogram extraction for feature comparison and future ML inputs.
  Commit: `Add mel spectrogram extraction`
- Added RMS envelope extraction for frame-by-frame loudness comparison.
  Commit: `Add RMS envelope extraction`
- Added single and multi-resolution STFT magnitude extraction for spectral comparison.
  Commit: `Add multi-resolution STFT features`
- Added spectral centroid extraction for frame-by-frame brightness comparison.
  Commit: `Add spectral centroid extraction`
- Added audio comparison helpers and a CLI script that reports current feature distances.
  Commit: `Add audio comparison script`
- Added the first weighted similarity distance for ranking candidate audio.
  Commit: `Add weighted similarity score`
- Added a ranking test proving identical audio scores better than different audio.
  Commit: `Test weighted similarity ranking`
- Completed Milestone C and moved the active phase to Milestone D search-based matching.
  Commit: `Complete Milestone C`
- Added normalized vector conversion from `SynthConfig` for search-based parameter optimization.
  Commit: `Add SynthConfig vector conversion`
- Added normalized vector reconstruction back into `SynthConfig`.
  Commit: `Add SynthConfig vector reconstruction`
- Added seeded random search over normalized synth parameter vectors.
  Commit: `Add random parameter search`
- Added search result export for saving the best patch JSON and rendered WAV into run directories.
  Commit: `Save best search candidate`
- Added optional progress callbacks and formatted progress output for search runs.
  Commit: `Add search progress output`
- Added a search regression test against a NeuroWave-generated target audio clip.
  Commit: `Test search against generated target`
- Added `report.json` export for saved search result directories.
  Commit: `Add search result report`
- Completed Milestone D and moved the active phase to Milestone E first ML model.
  Commit: `Complete Milestone D`
- Chose scikit-learn as the initial ML framework and documented why PyTorch is deferred.
  Commit: `Choose initial ML framework`
- Added a metadata-driven training dataset loader that returns audio feature inputs and normalized synth parameter targets.
  Commit: `Add training dataset loader`
- Added a scikit-learn MLP regression baseline for predicting normalized synth parameter vectors.
  Commit: `Add MLP regression baseline`
- Added a repeatable MLP training command and metrics path, then trained it on the local 10-example generated dataset.
  Commit: `Train MLP on small synthetic dataset`
- Added ignored `models/` checkpoint saving for the MLP training command.
  Commit: `Save MLP model checkpoints`
- Added a one-clip prediction script that loads a checkpoint and writes a predicted patch JSON.
  Commit: `Add audio patch prediction script`
- Added a prediction evaluation script that renders the predicted patch and writes feature-distance comparison reports.
  Commit: `Evaluate rendered ML predictions`
- Added optional local parameter-search refinement after ML prediction.
  Commit: `Add optional ML prediction refinement`
- Completed Milestone E; the next phase was later revised to scaled synthetic training before real-audio work.
  Commit: `Complete Milestone E`
- Revised the roadmap so Milestone F is scaled synthetic training before real-audio work, and added versioned dataset output support for `d2`.
  Commit: `Start scaled synthetic training milestone`
- Fixed random patch envelope scaling so generated training targets remain valid after normalization.
  Commit: `Keep random envelopes inside schema`
- Generated a local ignored `d2` dataset with 500 valid examples for scaled synthetic training.
  Commit: `Generate valid d2 dataset`
- Trained the MLP baseline on the local ignored `d2` dataset and saved the ignored checkpoint under `models/`.
  Commit: `Train MLP baseline on d2 dataset`
- Added metrics report output for training runs and saved the ignored `v2_sklearn_mlp_500seeds` metrics report under `runs/training/`.
  Commit: `Save v2 training metrics report`
- Added dataset-level evaluation and saved a 20-clip ignored `v2_sklearn_mlp_500seeds` on `d2` weighted-distance report under `runs/evaluation/`.
  Commit: `Evaluate v2 model across held-out clips`
- Compared `v1_sklearn_mlp_10seeds` and `v2_sklearn_mlp_500seeds` checkpoints on the same 20 `d2` clips; the larger model reduced mean and median weighted distance.
  Commit: `Compare v1 and v2 MLP evaluation reports`
- Decided to keep scikit-learn as the baseline while moving the next serious model track toward PyTorch spectrogram learning.
  Commit: `Decide next ML framework direction`
- Added the missing PyTorch spectrogram model milestone before real-audio work so the roadmap matches the ML decision.
  Commit: `Add PyTorch model milestone`
- Completed Milestone F and moved the active phase to Milestone G PyTorch spectrogram model work.
  Commit: `Complete Milestone F`
- Documented the PyTorch runtime decision: keep the current Python 3.14 sklearn environment stable and use a separate compatible runtime for PyTorch work.
  Commit: `Document PyTorch runtime decision`
- Added mel-spectrogram tensor export from generated metadata and saved local ignored `d2` tensors for future PyTorch training.
  Commit: `Export mel tensors for PyTorch training`
- Added explicit dataset/model/run naming rules and separated dataset IDs (`d1`, `d2`) from model IDs (`vN_modeltype_size`).
  Commit: `Clarify dataset and model naming`
- Updated the PyTorch runtime decision after verifying current official Python 3.14 support, installed `torch==2.12.0` into the project `.venv`, and verified local torch import.
  Commit: `Update PyTorch runtime decision`
- Added the first PyTorch CNN inverse model scaffold that maps mel-spectrogram tensors to normalized `SynthConfig` parameter vectors.
  Commit: `Add PyTorch inverse model`
- Added a PyTorch training script, trained `v3_pytorch_cnn_500seeds` on local ignored `d2` mel tensors, and saved ignored checkpoint plus training metrics report.
  Commit: `Train PyTorch inverse model on d2`
- Added one-clip PyTorch patch prediction from WAV input and verified it writes a renderable ignored JSON patch from the `v3_pytorch_cnn_500seeds` checkpoint.
  Commit: `Add PyTorch patch prediction`
- Added PyTorch prediction evaluation that renders the predicted patch, saves a WAV/report, and reports weighted audio distance for one target clip.
  Commit: `Evaluate PyTorch prediction audio`
- Added PyTorch dataset-level evaluation and saved an ignored 20-clip weighted-distance report for `v3_pytorch_cnn_500seeds` on `d2`.
  Commit: `Evaluate PyTorch model across d2`
- Compared `v3_pytorch_cnn_500seeds` against `v2_sklearn_mlp_500seeds` on `d2`; PyTorch improved mean weighted distance by `18.07`, median by `14.08`, and had zero failures.
  Commit: `Compare PyTorch and sklearn baselines`
- Decided to keep scikit-learn as a lightweight sanity baseline while making PyTorch the primary future model-development path.
  Commit: `Decide sklearn baseline status`
- Generated ignored `d3` with 10,000 seeds, exported mel tensors, trained `v4_pytorch_cnn_10kseeds`, and saved ignored training/evaluation reports. Parameter MAE improved, but weighted audio distance did not improve versus `v3`, so the next quality step should not be dataset size alone.
  Commit: `Train PyTorch model on 10k seeds`
- Replaced unsafe cloud multiprocessing changes with configurable worker counts, serial fallbacks, CPU thread limits, GPU-priority training device selection, and progress output for generation/export/training/evaluation.
  Commit: `Add safe multicore and GPU workflow`
- Reorganized `README.md` into a clear setup, dataset generation, PyTorch training, evaluation, and comparison workflow. Added CUDA-specific PyTorch requirements for future NVIDIA training machines.
  Commit: `Reorganize training README`
- Added PyTorch per-parameter metrics, waveform accuracy metrics, optional fixed benchmark splits, and comparison reporting that combines parameter metrics with rendered-audio evaluation.
  Commit: `Add PyTorch model diagnostics`
- Decided the next inverse-model design should not predict `freq`; synthetic patch `freq` should condition the model, real-audio pitch should later come from classical pitch estimation or manual input, and `length` should remain visible because it affects ADSR interpretation.
  Commit: `Document pitch conditioning strategy`
- Completed Milestone G tracker closeout and moved the active phase to Milestone H model capability work.
  Commit: `Complete Milestone G`
- Added grouped PyTorch target metrics for pitch, duration, pitch-conditioned timbre, oscillator, filter, and ADSR error reporting.
  Commit: `Add grouped model metrics`
- Replaced the default PyTorch waveform target path with classification heads while keeping scalar waveform regression available for legacy checkpoint comparison. Scaling note: move from waveform classification to continuous wave-mix targets when the synth schema exposes wave mixtures as first-class parameters.
  Commit: `Add waveform classification heads`
- Added `pitch_conditioned_timbre` training mode that appends exact synthetic pitch as an input channel and removes `freq` from the output target. Scaling note: replace the broadcast pitch channel with a cleaner metadata/context embedding when the model family grows beyond simple CNN inputs.
  Commit: `Add pitch conditioned target mode`
- Confirmed `length` remains visible to the model design because duration changes ADSR interpretation and pluck/key/pad behavior. Scaling note: revisit whether `length` should become input context, output target, or both after the first pitch-conditioned capability model reports grouped ADSR and duration metrics.
  Commit: `Document length modeling decision`
- Added parameter-weighted PyTorch loss presets, including an `audibility` preset that emphasizes waveform, detune, filter, and envelope parameters while keeping `flat` as the default. Scaling note: tune or learn these weights from rendered-audio evaluation once enough benchmark reports exist.
  Commit: `Add parameter weighted loss`
- Added PyTorch optimizer and training controls for AdamW, weight decay, step LR scheduling, early stopping, validation-loss history, and best-validation checkpoint selection. Scaling note: add richer schedulers such as cosine or reduce-on-plateau after the first medium/large model comparisons.
  Commit: `Add training control options`
- Added named PyTorch model-size presets (`small`, `medium`, `large`) so capacity can scale without editing training code. Scaling note: add residual blocks or attention-style encoders after comparing these preset sizes on the fixed benchmark.
  Commit: `Add scalable model sizes`
- Added `time_frequency` pooling mode that preserves a small time-frequency grid before the prediction head, while keeping `global` pooling for legacy comparisons. Scaling note: evaluate whether the retained grid should grow or become attention-based once medium and large models are benchmarked.
  Commit: `Preserve time frequency pooling`
- Added rendered-audio evaluation diagnostics that save target patches, predicted patches, normalized per-parameter errors, and the worst clips ranked by weighted audio distance. Scaling note: use these diagnostics to design hybrid losses and model heads before increasing dataset size again.
  Commit: `Add worst clip evaluation diagnostics`
- Added a `hybrid` PyTorch loss preset for v2.1 that moderately emphasizes waveform identity, oscillator levels, detune, resonance, sustain, and release while keeping cutoff closer to flat weighting.
  Commit: `Add hybrid loss preset`
- Reexamined the model-capability roadmap after `v10`, `v2.0`, and `v2.1`. The current best rendered-audio setup is still `v10`, while `v2.1` has the best parameter metrics but worse audio due to filter/cutoff failures. Conclusion: stop minor loss-only tuning and move the next major model series to `v3.0` with prediction-spread diagnostics, safer render validation, and group-balanced multi-head continuous prediction.
  Commit: `Reassess model capability roadmap`
- Implemented the v3.0-ready model setup: grouped continuous heads, group-balanced loss, prediction distribution diagnostics, checkpointed head-mode metadata, safer failed-render reports, portable JSON path serialization, and updated training commands/naming rules.
  Commit: `Implement v3 model setup`
