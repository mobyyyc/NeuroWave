# MiniSynth Progress Tracker

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

Current active phase: Milestone A - Clean Synth Engine.

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
- [ ] Add tests for wave-mix normalization.
- [ ] Move `lowpass_filter()` into `minisynth/filters.py`.
- [ ] Add tests that low-pass filtering preserves array length and finite values.
- [ ] Move `render_patch()` into `minisynth/engine.py`.
- [ ] Make `render_patch()` return audio only and never save files or show plots.
- [ ] Add tests for deterministic rendering from the same patch.
- [ ] Create initial `SynthConfig` data model in `minisynth/schema.py`.
- [ ] Define parameter metadata type with name, kind, min, max, default, scale, group, and `ml_enabled`.
- [ ] Add linear normalization and denormalization helpers.
- [ ] Add logarithmic normalization and denormalization helpers.
- [ ] Add tests for normalization round trips.
- [ ] Convert `PATCHES["dark_saw"]` into `presets/dark_saw.json`.
- [ ] Add `minisynth/io.py` helpers to load and save JSON patches.
- [ ] Add `scripts/render_patch.py` that renders a preset JSON to a WAV path.
- [ ] Update `README.md` with setup and render commands.
- [ ] Keep old `synth.py` as a compatibility wrapper or remove it after scripts replace it.
- [ ] Run the new render script and verify it recreates `dark_saw.wav`.
- [ ] Commit Milestone A completion.

## Milestone B: Dataset Tools

Goal: generate labeled synthetic data from known parameters.

- [ ] Create `scripts/random_patch.py` with seeded random patch generation.
- [ ] Add constraints so random patches are not silent.
- [ ] Add constraints so generated audio avoids clipping.
- [ ] Save generated patch JSON files under `data/generated/v1/params/`.
- [ ] Save generated WAV files under `data/generated/v1/audio/`.
- [ ] Write `metadata.jsonl` for generated clips.
- [ ] Add a small sample generation command to `README.md`.
- [ ] Generate a tiny local dataset of 10 patches for manual inspection.
- [ ] Add tests for reproducible random patch generation from a seed.
- [ ] Commit Milestone B completion.

## Milestone C: Audio Features And Similarity

Goal: compare target audio and rendered audio in a useful, phase-tolerant way.

- [ ] Add mono conversion helper.
- [ ] Add resampling helper.
- [ ] Add loudness normalization helper.
- [ ] Add mel spectrogram extraction in `minisynth/features.py`.
- [ ] Add RMS envelope extraction.
- [ ] Add multi-resolution STFT magnitude extraction.
- [ ] Add spectral centroid extraction.
- [ ] Add `scripts/compare_audio.py`.
- [ ] Define first weighted similarity score.
- [ ] Test that identical audio scores better than different audio.
- [ ] Commit Milestone C completion.

## Milestone D: Search-Based Matching

Goal: match target audio by optimizing synth parameters before training neural networks.

- [ ] Add parameter vector to `SynthConfig` conversion.
- [ ] Add vector to `SynthConfig` reconstruction.
- [ ] Add random search over parameter vectors.
- [ ] Save best candidate patch and WAV into `runs/`.
- [ ] Add progress output for search runs.
- [ ] Test matching against a MiniSynth-generated target.
- [ ] Add a simple comparison report file in each run directory.
- [ ] Commit Milestone D completion.

## Milestone E: First ML Model

Goal: predict synth parameters from audio features on synthetic data.

- [ ] Choose initial ML framework.
- [ ] Create dataset loader for generated metadata.
- [ ] Create first CNN plus MLP model.
- [ ] Train on a small synthetic dataset.
- [ ] Save model checkpoints under `models/`.
- [ ] Add prediction script for one audio clip.
- [ ] Render predicted patch and compare to target.
- [ ] Add optional optimizer refinement after prediction.
- [ ] Commit Milestone E completion.

## Milestone F: Real Audio Prototype

Goal: approximate clean single-note real audio clips.

- [ ] Add target audio import script.
- [ ] Add onset detection and trimming.
- [ ] Add pitch estimation or manual pitch input.
- [ ] Add real-clip prediction pipeline.
- [ ] Export target/result comparison WAVs.
- [ ] Export comparison feature plots or data.
- [ ] Document known failure cases.
- [ ] Commit Milestone F completion.

## Milestone G: Interface And Workflow

Goal: make MiniSynth usable as a tool.

- [ ] Define CLI command names.
- [ ] Add CLI entry point.
- [ ] Add `minisynth render`.
- [ ] Add `minisynth random`.
- [ ] Add `minisynth compare`.
- [ ] Add `minisynth match`.
- [ ] Add `minisynth predict`.
- [ ] Consider a simple local web UI after CLI is stable.
- [ ] Commit Milestone G completion.

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
