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

Current active phase: Milestone F - Scaled Synthetic Training.

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
- [x] Save generated patch JSON files under `data/generated/v1/params/`.
- [x] Save generated WAV files under `data/generated/v1/audio/`.
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

- [x] Add versioned dataset output support for `data/generated/v2/`.
- [x] Fix random patch envelope constraints so generated targets stay inside schema ranges.
- [ ] Generate a local ignored `v2` dataset with 500 examples.
- [ ] Train the MLP baseline on the `v2` dataset.
- [ ] Save a metrics report for the `v2` training run.
- [ ] Add evaluation across multiple held-out synthetic clips.
- [ ] Compare the `v1` tiny model against the `v2` larger model.
- [ ] Decide whether to keep scikit-learn MLP or move next to PyTorch.
- [ ] Commit Milestone F completion.

## Milestone G: Real Audio Prototype

Goal: approximate clean single-note real audio clips.

- [ ] Add target audio import script.
- [ ] Add onset detection and trimming.
- [ ] Add pitch estimation or manual pitch input.
- [ ] Add real-clip prediction pipeline.
- [ ] Export target/result comparison WAVs.
- [ ] Export comparison feature plots or data.
- [ ] Document known failure cases.
- [ ] Commit Milestone G completion.

## Milestone H: Interface And Workflow

Goal: make NeuroWave usable as a tool.

- [ ] Define CLI command names.
- [ ] Add CLI entry point.
- [ ] Add `neurowave render`.
- [ ] Add `neurowave random`.
- [ ] Add `neurowave compare`.
- [ ] Add `neurowave match`.
- [ ] Add `neurowave predict`.
- [ ] Consider a simple local web UI after CLI is stable.
- [ ] Commit Milestone H completion.

## Progress Log

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
- Added dataset helpers and CLI support for saving generated patch JSON files under `data/generated/v1/params/`.
  Commit: `Save generated patch params`
- Added dataset helpers and CLI support for saving generated WAV files under `data/generated/v1/audio/`.
  Commit: `Save generated dataset audio`
- Added `metadata.jsonl` generation so each dataset row links seed, patch JSON, and rendered WAV paths.
  Commit: `Write generated dataset metadata`
- Documented the small dataset generation command in the README.
  Commit: `Document dataset generation command`
- Generated the local ignored 10-example dataset under `data/generated/v1/` for manual inspection.
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
- Revised the roadmap so Milestone F is scaled synthetic training before real-audio work, and added versioned dataset output support for `v2`.
  Commit: `Start scaled synthetic training milestone`
- Fixed random patch envelope scaling so generated training targets remain valid after normalization.
  Commit: `Keep random envelopes inside schema`
