# NeuroWave Roadmap

## Goal

NeuroWave is intended to become a synthesizer that can listen to a short audio example and recreate a similar sound by choosing synthesizer parameters.

Branding note: the project is now called NeuroWave. The current repository folder and Python package remain `MiniSynth` / `minisynth` until an explicit architecture rename is requested.

The core idea is analysis by synthesis:

1. Build a deterministic synthesizer whose entire sound is controlled by explicit parameters.
2. Render many sounds from known parameter values.
3. Train or optimize a system that maps target audio back to those parameter values.
4. Use the predicted parameters to render a new sound.
5. Compare the rendered sound against the target and refine.

The AI should not directly generate arbitrary audio at first. It should learn to control the synthesizer. That keeps the output editable, explainable, repeatable, and useful as a real instrument.

## Core Design Principle

Every sound-changing decision must be represented as a parameter.

Good parameters are:

- Explicit: every value has a name and clear meaning.
- Bounded: every value has a min, max, default, and valid type.
- Normalizable: every value can be converted to and from a 0 to 1 ML-friendly range.
- Deterministic: the same parameter config always renders the same audio.
- Serializable: a full patch can be saved as JSON.
- Stable: old patches should remain loadable when new parameters are added.

The machine learning model should predict a structured `SynthConfig`, not hidden state inside the synth code.

## Stage 1: Build The Parameterized Synthesizer

### Objective

Replace the current script-style synth with a small synth engine that can render audio from a complete parameter object.

The current synth already has useful building blocks:

- Sample rate.
- Oscillators.
- Detuning.
- Low-pass filter.
- ADSR envelope.
- WAV rendering.
- Spectrogram visualization.

The next version should make those pieces schema-driven.

### Proposed File Structure

```text
NeuroWave/
  minisynth/
    __init__.py
    engine.py          # render audio from SynthConfig
    schema.py          # parameter definitions and normalization
    oscillators.py     # waveform generation
    filters.py         # filters
    envelopes.py       # ADSR and future envelopes
    effects.py         # optional effects
    features.py        # spectrogram and audio feature extraction
    io.py              # load/save patches and audio
  presets/
    dark_saw.json
  scripts/
    render_patch.py
    random_patch.py
    compare_audio.py
  tests/
    test_schema.py
    test_render.py
  README.md
  PLAN.md
```

The exact structure can change, but the key separation should stay:

- Engine code renders sound.
- Schema code defines valid parameters.
- Scripts call the engine.
- Presets are data files, not hardcoded Python dictionaries.

### SynthConfig V1

Use a fixed-size config at first. Fixed-size parameter vectors are much easier for ML than variable numbers of modules.

Recommended V1 modules:

- Global parameters.
- Three oscillators.
- One noise source.
- One mixer.
- One multi-mode filter.
- One amplifier ADSR envelope.
- One filter ADSR envelope.
- Two LFOs.
- Basic effects.

Even if an oscillator or LFO is unused, it still exists in the config. It can be disabled with a level or amount value of 0.

### Example Config Shape

```json
{
  "version": 1,
  "global": {
    "sample_rate": 44100,
    "note_midi": 60,
    "duration": 1.5,
    "gain": 0.8
  },
  "oscillators": [
    {
      "enabled": 1.0,
      "level": 0.8,
      "pan": 0.0,
      "detune_cents": 0.0,
      "phase": 0.0,
      "wave_mix": {
        "sine": 0.0,
        "triangle": 0.0,
        "saw": 1.0,
        "square": 0.0,
        "noise": 0.0
      },
      "pulse_width": 0.5
    }
  ],
  "filter": {
    "enabled": 1.0,
    "type": "lowpass",
    "cutoff_hz": 1200.0,
    "resonance": 0.7,
    "drive": 0.0,
    "env_amount": 0.0,
    "key_tracking": 0.0
  },
  "amp_env": {
    "attack": 0.01,
    "decay": 0.2,
    "sustain": 0.7,
    "release": 0.3
  },
  "filter_env": {
    "attack": 0.01,
    "decay": 0.2,
    "sustain": 0.5,
    "release": 0.2
  },
  "effects": {
    "chorus_amount": 0.0,
    "delay_mix": 0.0,
    "reverb_mix": 0.0
  }
}
```

This example is larger than the current synth, but it shows the target direction.

### Wave Shape Parameter Design

Avoid a single categorical `wave = "saw"` parameter as the long-term representation.

For ML, use a continuous wave mixture:

```json
"wave_mix": {
  "sine": 0.0,
  "triangle": 0.2,
  "saw": 0.7,
  "square": 0.1,
  "noise": 0.0
}
```

The synth renders each base waveform and blends them according to these weights.

Rules:

- Each weight is between 0 and 1.
- The weights should sum to 1, or be normalized internally.
- This allows the model to predict sounds between classic waveforms.
- This is easier to learn than choosing one hard waveform class.

Later, wave shape can become more powerful with wavetable position, harmonic controls, or additive partial levels. The first ML-friendly version should stay simple.

### Parameter Schema

Every parameter should be registered in a schema.

Each schema entry should include:

- Name, such as `osc1.level`.
- Type, such as float, int, bool, or enum.
- Minimum value.
- Maximum value.
- Default value.
- Scaling, such as linear, logarithmic, or categorical.
- Group, such as oscillator, filter, envelope, or effects.
- Whether the parameter is predicted by ML.

Example:

```python
Parameter(
    name="filter.cutoff_hz",
    kind="float",
    minimum=20.0,
    maximum=20000.0,
    default=1200.0,
    scale="log",
    ml_enabled=True,
)
```

Important scaling choices:

- Frequency should usually be logarithmic.
- Detune should usually be linear in cents.
- Levels should usually be linear or perceptual depending on use.
- Times should often be logarithmic, because 0.005s to 0.05s matters a lot.
- Wave mix should use normalized weights.

### Render Contract

The main renderer should have a stable contract:

```python
audio = render(config: SynthConfig) -> np.ndarray
```

It should:

- Return floating point audio in a known range, probably -1.0 to 1.0.
- Never write files directly.
- Never open plots directly.
- Never rely on global patch dictionaries.
- Avoid random behavior unless a seed is part of the config.

Separate scripts can save WAV files, show spectrograms, or play audio.

### Stage 1 Acceptance Criteria

Stage 1 is complete when:

- A patch can be saved and loaded as JSON.
- A patch can be converted to a normalized ML vector and back.
- The existing `dark_saw` sound can be represented as a JSON preset.
- Rendering is deterministic.
- The renderer can run without opening a plot window.
- Unit tests cover schema normalization and basic rendering.

## Stage 2: Dataset Generation

### Objective

Generate a large synthetic dataset of parameter configs and rendered audio clips.

Because the synth creates the data, every audio file has exact ground-truth parameters. This makes supervised learning possible.

### Dataset Format

Recommended layout:

```text
data/
  generated/
    v1/
      metadata.jsonl
      audio/
        000000.wav
        000001.wav
      params/
        000000.json
        000001.json
      features/
        000000.npz
        000001.npz
```

Each metadata row should include:

- ID.
- Audio path.
- Parameter path.
- Feature path.
- Synth version.
- Random seed.
- Note MIDI.
- Duration.
- Loudness summary.

Generated data should usually be ignored by git. Small example presets can be committed.

### Parameter Sampling

Do not sample every parameter uniformly at random without constraints. That will create many useless or silent sounds.

Use a constrained random sampler:

- Ensure at least one oscillator has meaningful level.
- Keep total gain under control.
- Avoid extreme attack and release values in early datasets.
- Avoid filter cutoff values that make most sounds silent.
- Keep resonance below unstable ranges.
- Sample some simple patches and some complex patches.
- Include known hand-designed presets as anchors.

Sampling should support curriculum stages:

1. Simple one-oscillator sounds.
2. Two-oscillator detuned sounds.
3. Filtered sounds.
4. Envelope variation.
5. LFO and effects variation.

This lets ML learn the synth in layers instead of all at once.

### Stage 2 Acceptance Criteria

Stage 2 is complete when:

- A script can generate N random patches and WAV files.
- Each generated clip has matching parameter JSON.
- Dataset generation is reproducible from a seed.
- Generated audio is normalized consistently.
- A small dataset can be produced locally.

## Stage 3: Audio Feature And Loss Pipeline

### Objective

Create a way to compare a target sound with a rendered sound.

Raw waveform difference is usually not enough. Two sounds can be perceptually similar but have different phase. The comparison should focus on spectral and envelope similarity.

### Feature Extraction

Recommended features:

- Mono audio at a fixed sample rate.
- Trimmed or aligned onset.
- RMS envelope.
- Mel spectrogram.
- Multi-resolution STFT magnitude.
- Spectral centroid.
- Spectral rolloff.
- Spectral flatness.
- Optional pitch estimate.

For early work, assume:

- Single note.
- Known or estimated pitch.
- Short clip, around 1 to 2 seconds.
- No complex melody.
- Minimal background noise.

### Loss Function

Use a weighted loss:

```text
loss =
  mel_spectrogram_loss
  + multi_resolution_stft_loss
  + envelope_loss
  + loudness_loss
  + optional_pitch_loss
```

This same loss can be used for:

- Search-based matching.
- ML evaluation.
- Fine-tuning or refinement.

### Stage 3 Acceptance Criteria

Stage 3 is complete when:

- The system can compare two audio clips and return a similarity score.
- Similar patches score closer than very different patches.
- The loss is insensitive enough to small phase shifts.
- Feature extraction is deterministic.

## Stage 4: Matching Without Neural Networks

### Objective

Before training an ML model, build a baseline optimizer that searches for synth parameters matching a target audio clip.

This is important because:

- It proves the synth parameter space can recreate useful sounds.
- It creates a benchmark for future ML.
- It can refine ML predictions later.
- It works before a large dataset exists.

### Search Methods

Start simple:

- Random search.
- Coarse grid search over important parameters.
- Coordinate descent.

Then use stronger black-box optimizers:

- CMA-ES.
- Differential evolution.
- Bayesian optimization.

The optimizer loop:

1. Load target audio.
2. Extract target features.
3. Sample or mutate parameter vectors.
4. Render each candidate.
5. Extract candidate features.
6. Compute loss.
7. Keep the best candidate.
8. Save best patch JSON and rendered WAV.

### Stage 4 Acceptance Criteria

Stage 4 is complete when:

- Given a target made by NeuroWave, the optimizer can recover a similar patch.
- The recovered patch produces a clearly similar sound.
- The result is saved as JSON plus WAV.
- The optimizer can be used as a quality baseline for ML.

## Stage 5: Supervised Inverse Model

### Objective

Train a model that predicts synthesizer parameters from audio features.

Input:

- Mel spectrogram or multi-channel spectral features from a target clip.

Output:

- Normalized parameter vector for `SynthConfig`.

### Model Shape

Start with a modest model:

- CNN over mel spectrogram.
- Small MLP head for parameter output.
- Sigmoid outputs for bounded continuous parameters.
- Softmax outputs for wave mix groups.
- Separate heads for categorical parameters if enums are used.

Avoid starting with a huge model. The synth and dataset will likely change often.

### Training Loss

Use a combination of:

- Parameter loss: predicted normalized vector vs ground truth vector.
- Audio reconstruction loss: render predicted patch and compare features to target.

At first, training can use parameter loss only. Later, include audio loss.

If direct rendering inside training is too slow or not differentiable, use the audio loss only for evaluation and refinement.

### Prediction Pipeline

1. Load target audio.
2. Normalize, trim, and resample.
3. Extract features.
4. Model predicts normalized parameter vector.
5. Convert vector to `SynthConfig`.
6. Render predicted patch.
7. Optionally refine with Stage 4 optimizer.
8. Save patch and audio.

### Stage 5 Acceptance Criteria

Stage 5 is complete when:

- The model predicts valid synth configs for unseen synthetic audio.
- Predictions beat random search at equal compute.
- Predicted patches are editable and render correctly.
- Optional optimizer refinement improves the model output.

## Stage 6: Real Audio Clips

### Objective

Use short user-provided audio clips, not only synthetic NeuroWave clips.

This is harder because real audio may contain:

- Background noise.
- Reverb.
- Compression.
- Multiple notes.
- Pitch drift.
- Instruments outside the synth's possible parameter space.

### Early Constraints

Limit the first real-audio version to:

- One note.
- One main instrument sound.
- Clean recording.
- 1 to 2 seconds.
- Known or estimated pitch.

The system should explain when a target is outside the synth's current capability.

### Real Clip Pipeline

1. Load clip.
2. Convert to mono.
3. Resample.
4. Detect onset and trim.
5. Estimate pitch.
6. Normalize loudness.
7. Run inverse model.
8. Refine with optimizer.
9. Export patch and rendered comparison WAV.

### Stage 6 Acceptance Criteria

Stage 6 is complete when:

- A clean single-note external sample can be approximated.
- The system saves the estimated patch.
- The system exports a before/after comparison.
- Failure cases are clear instead of silent.

## Stage 7: Interface And Workflow

### Objective

Make NeuroWave usable as a tool, not just a research script.

Possible interfaces:

- CLI first.
- Simple local web UI later.
- Optional MIDI input later.

Useful commands:

```bash
neurowave render presets/dark_saw.json --out dark_saw.wav
neurowave random --count 100 --out data/generated/v1
neurowave match target.wav --out runs/match_001
neurowave train --dataset data/generated/v1
neurowave predict target.wav --model models/inverse_v1.pt
```

### UI Ideas

The UI should show:

- Target spectrogram.
- Rendered spectrogram.
- Difference score.
- Predicted parameters.
- Audio playback for target and result.
- Manual parameter editing.
- Re-run refinement button.

## Suggested Milestones

### Milestone A: Clean Synth Engine

- Convert `synth.py` into modular engine code.
- Add `SynthConfig`.
- Add schema normalization.
- Convert `dark_saw` to JSON.
- Add tests.

### Milestone B: Dataset Tools

- Add random patch generator.
- Add batch renderer.
- Add metadata JSONL.
- Generate first small dataset.

### Milestone C: Audio Features And Similarity

- Add feature extraction.
- Add audio similarity loss.
- Compare target and candidate WAV files.
- Prove identical audio scores better than different audio.

### Milestone D: Search-Based Matching

- Add parameter-vector search over `SynthConfig`.
- Save best candidate patches and WAV files under `runs/`.
- Match NeuroWave-generated targets before neural training.
- Use search as a future refinement path for ML predictions.

### Milestone E: First ML Model

- Start with a scikit-learn regression baseline using existing local dependencies.
- Predict normalized synth parameter vectors from extracted audio features.
- Evaluate on synthetic validation data.
- Move to PyTorch and CNN-style spectrogram models after the baseline data path is proven.
- Refine predictions with optimizer.

### Milestone F: Scaled Synthetic Training

- Generate larger versioned synthetic datasets, starting around 500 examples and scaling upward.
- Train the baseline model on larger datasets and save metrics for each run.
- Evaluate against multiple held-out synthetic clips, not only one target WAV.
- Compare larger-dataset models against the tiny 10-example prototype.
- Decide whether the scikit-learn MLP is still useful or whether the next model should move to PyTorch.

### Milestone G: PyTorch Spectrogram Model

- Make an explicit PyTorch dependency and runtime decision.
- Create mel-spectrogram dataset tensors from generated metadata.
- Build the first PyTorch inverse model that predicts normalized `SynthConfig` vectors.
- Train on v2 or a larger synthetic dataset.
- Save PyTorch checkpoints and training metrics.
- Predict a patch from one clip and render it.
- Evaluate across synthetic dataset clips.
- Compare PyTorch directly against the scikit-learn baseline.
- Decide whether scikit-learn remains as a lightweight baseline or can be removed.

### Milestone H: Model Capability And Target Quality

Goal: make the inverse model itself strong enough that it is not the limiting factor when trained on a good synthetic dataset.

Evidence from local PyTorch reports:

- `v5_pytorch_cnn_10kseeds` reached `test_loss = 0.0461` and `test_mae = 0.1616`.
- `v8_pytorch_cnn_50kseeds` reached `test_loss = 0.0375` and `test_mae = 0.1339`.
- `v9_pytorch_cnn_200kseeds` reached `test_loss = 0.0356` and `test_mae = 0.1284`.
- Train and test metrics remain close, so the current issue is not mainly overfitting.
- More data helps, but the current small CNN, scalar categorical encoding, and plain averaged MSE are unlikely to reach `test_mae = 0.05` by scale alone.

Roadmap:

- Add per-parameter validation metrics so average MAE cannot hide weak targets.
- Report waveform accuracy separately from continuous-parameter MAE.
- Replace scalar waveform enum regression with explicit classification heads or continuous wave-mix targets.
- Remove `freq` from the core timbre prediction target while still giving the model pitch context. For synthetic data, use the exact patch `freq` as known conditioning; for real audio later, estimate pitch with a classical monophonic pitch detector or allow manual pitch input.
- Keep `length` in the model design for now because note duration changes the interpretation of ADSR parameters and the difference between plucks and pads.
- Add target groups so reports can separate pitch-conditioned timbre quality from pitch estimation and duration handling.
- Add parameter-weighted losses so perceptually important parameters are not treated the same as low-impact parameters.
- Build a larger model family with enough capacity for the current task, such as deeper CNN blocks, residual blocks, wider channel counts, and stronger MLP heads.
- Preserve time-frequency structure longer instead of collapsing the spectrogram too early with global average pooling.
- Add regularization and optimizer controls: AdamW, weight decay, learning-rate scheduling, early stopping, and best-validation checkpoint saving.
- Add repeatable train/evaluate experiment configs so model comparisons differ by one intentional variable at a time.
- Add a fixed benchmark evaluation set that is never used for training or hyperparameter selection.
- Track both parameter metrics and rendered-audio metrics for every serious model.
- Evaluate whether model predictions are better than nearest-neighbor retrieval and random/search baselines at similar compute.
- Add optional prediction refinement after the model only after raw model quality is measured clearly.

Reassessment after the first pitch-conditioned capability runs:

- `v10_pytorch_cnn_pitchctx_weighted_medium_tfpool_50kseeds` gave the best rendered-audio distance among the first capability models, but it had weaker parameter metrics and weak waveform accuracy.
- `v2.0_pytorch_cnn_pitchctx_flat_medium_tfpool_50kseeds` improved parameter MAE and waveform accuracy, but rendered audio got worse.
- `v2.1_pytorch_cnn_pitchctx_hybrid_medium_tfpool_50kseeds` improved test MAE, waveform accuracy, oscillator MAE, and ADSR MAE again, but rendered audio got worse because filter/cutoff errors grew.
- Prediction diagnostics show strong regression toward average parameter values, especially for oscillator levels, resonance, ADSR times, release, and sometimes length. This means the current shared continuous head and scalar loss tuning are likely a model-design bottleneck.

The next major model series should therefore be `v3.0`, not another minor loss-weight tweak.
The preferred setup is:

- Keep pitch-conditioned timbre prediction: exact synthetic `freq` is input context and not an output target.
- Keep waveform classification for the current schema, but treat continuous wave-mix targets as the next synth-schema upgrade if waveform accuracy remains limiting.
- Replace the single shared continuous output vector with group-aware or per-parameter heads, such as separate heads for length, oscillator levels/detune, filter cutoff/resonance, and ADSR.
- Use group-balanced loss so oscillator, filter, ADSR, and duration objectives cannot drown each other out.
- Track prediction distribution diagnostics, especially target-vs-predicted standard deviation per parameter, so mean-collapse is visible before rendered-audio evaluation.
- Use a larger or residual CNN backbone only after the reporting can prove whether capacity, target representation, or data ambiguity is the bottleneck.

Acceptance criteria:

- Training reports include per-parameter MAE, continuous-parameter MAE, waveform accuracy, train/test loss, train/test MAE, dataset ID, model ID, epochs, batch size, learning rate, optimizer, scheduler, and checkpoint-selection rule.
- Training reports include grouped metrics such as timbre MAE, ADSR MAE, oscillator MAE, filter MAE, waveform accuracy, and pitch-conditioned metrics that exclude `freq` from model quality.
- The model architecture can intentionally be scaled without editing core training code.
- At least one stronger model reaches materially better validation quality than `v9_pytorch_cnn_200kseeds`.
- The project can distinguish model bottlenecks from data bottlenecks using benchmark reports.
- The target benchmark for this phase is `test_mae <= 0.05` on a well-defined synthetic holdout set, or a documented explanation of which parameters make that target unrealistic under the current synth representation.

Reassessment after `v3.0` and `v3.1`:

- `v3.0_restructure` proved the pitch-conditioned grouped-head design was the right
  direction: parameter MAE, waveform accuracy, and rendered-audio distance all improved
  together.
- `v3.1_500ksamples` proved the same design scales with data: test MAE reached about
  `0.0715`, waveform accuracy reached about `0.882`, and the 200-clip d8 mean weighted
  distance dropped to about `32.2`.
- The remaining dominant error is oscillator mix quality, especially `osc1_level`,
  `osc2_level`, and `osc2_detune`. The level predictions still regress toward average
  levels, even though waveform classification is now strong.
- The two oscillator slots are partly exchangeable. A patch with oscillator 1 as a quiet
  saw and oscillator 2 as a loud sine can be perceptually equivalent, or nearly equivalent,
  to the same waves and levels assigned to the opposite slots. Treating `osc1_level` and
  `osc2_level` as independent slot-specific regression targets therefore creates an
  artificial target ambiguity.

The next model series should therefore be `v3.2`, focused on canonical oscillator-mix
targets rather than a larger backbone. The preferred design is:

- Convert oscillator slots into a canonical order before training targets are produced.
  A deterministic rule should sort the two oscillators by waveform identity and/or level
  so equivalent two-oscillator patches map to the same target representation.
- Predict oscillator mix in perceptual terms: waveform identity plus level associated
  with that waveform, not arbitrary slot identity.
- Consider replacing raw `osc1_level` and `osc2_level` targets with `osc_total_level`
  and `osc_balance`, then reconstruct render parameters after prediction.
- Keep `osc2_detune` tied to the canonical pair definition. If oscillator order changes,
  detune sign and meaning must be handled explicitly so the rendered result remains
  equivalent.
- Add slot-invariant oscillator diagnostics: best assignment error, total-level error,
  balance error, per-wave level error, and worst-clip analysis focused on oscillator mix.
- Keep the rest of the proven v3 setup unchanged for the first v3.2 run: pitch-conditioned
  timbre, waveform classification, large CNN, time-frequency pooling, grouped heads,
  group-balanced loss, AdamW, early stopping, and best-validation checkpointing.

Acceptance criteria for `v3.2`:

- Reports show oscillator-level errors in a representation that is not punished merely
  because the two oscillator slots were swapped.
- Reconstructed `osc1_level` and `osc2_level` MAE improve meaningfully versus v3.1's
  roughly `0.18` per level, or the report proves the rendered-audio metric improves even
  when slot-specific MAE is not directly comparable.
- Oscillator grouped MAE improves versus v3.1's roughly `0.111`.
- The fixed d8 evaluation mean weighted distance moves below v3.1's roughly `32`, with
  no regression in median distance or waveform accuracy.

Reassessment after `v3.2`:

- `v3.2_oscmix` improved training quality strongly: test MAE reached about `0.0527`,
  waveform accuracy reached about `0.969`, and oscillator grouped MAE improved versus
  v3.1. This confirms oscillator slot ambiguity was a real model bottleneck.
- The fair d8 rendered-audio comparison did not improve median distance:
  `v3.2_oscmix` reduced worst-case failures but increased the 1000-clip median distance.
- The likely cause is that canonical oscillator swapping is not fully compatible with
  the current synth's detune convention. `osc2_detune` is relative to the main/base
  oscillator, so swapping oscillator roles changes the meaning of the detune target.

The next model series should therefore be `v3.3`, focused on main/detuned oscillator
roles rather than fully slot-canonical oscillator roles. The preferred design is:

- Treat the known pitch context as the main oscillator's base frequency.
- Keep `osc1` as `main_wave`, the waveform heard at the supplied/base frequency.
- Keep `osc2` as `detuned_wave`, the second waveform heard at a relative pitch offset.
- Replace raw `osc1_level` and `osc2_level` with `osc_total_level` and
  `detuned_balance`, then reconstruct slot levels for rendering.
- Replace the output name `osc2_detune` with `detune_amount` so reports express the
  learned target as relative pitch from the known main frequency.
- Add main/detuned diagnostics: main wave error, detuned wave error, total-level error,
  detuned-balance error, and normalized detune error.
- Keep the proven v3 architecture unchanged: pitch-conditioned input, waveform
  classification, large CNN, time-frequency pooling, grouped heads, group-balanced
  loss, AdamW, early stopping, and best-validation checkpointing.

Acceptance criteria for `v3.3`:

- Training MAE stays near or below v3.2's roughly `0.0527`.
- The fixed d8 1000-clip median weighted distance improves versus v3.2's roughly
  `16.54`, with no major regression in mean distance.
- Detune-heavy worst clips show lower normalized detune error or clearly improved
  rendered distance.

Reassessment after `v3.3`:

- `v3.3_main_detuned_mix` improved the fixed d8 rendered-audio benchmark versus both
  v3.1 and v3.2 on mean and median distance. This confirms the main/detuned target
  representation is the right model setup.
- v3.3 did not beat v3.2 on abstract parameter MAE, which confirms that raw MAE is no
  longer the deciding metric for this phase.
- Remaining failures are concentrated in audible oscillator mistakes: wrong noisy or
  detuned waveform when the oscillator is loud, large detune errors when the detuned
  oscillator is audible, and total-level overprediction on quiet targets.

The next model series should therefore be `v3.4`, focused on audibility-aware loss while
keeping the v3.3 target representation and architecture unchanged. The preferred design is:

- Keep `main_detuned_mix` targets.
- Keep the grouped-head large CNN architecture, pitch context, waveform classification,
  AdamW, step scheduler, early stopping, and best-validation checkpointing.
- Weight `main_wave` classification more when the main oscillator target level is high.
- Weight `detuned_wave` classification and `detune_amount` regression more when the
  detuned oscillator target level is high.
- Weight `detuned_balance` by total oscillator audibility.
- Penalize oscillator total-level overshoot more when the target total level is quiet, to
  reduce rare loud hallucination outliers.

Acceptance criteria for `v3.4`:

- The fixed d8 1000-clip mean weighted distance improves versus v3.3's roughly `30.38`.
- The fixed d8 1000-clip median stays at or below v3.3's roughly `10.48`.
- The worst-case max distance improves versus v3.3's roughly `1056.85`, or worst-clip
  diagnostics show the remaining max failure is outside oscillator audibility handling.

Reassessment after `v3.4`:

- `v3.4_audible_loss` became the best rendered-audio model so far on the fixed d8
  1000-clip evaluation, improving mean and median weighted distance versus v3.3.
- The remaining worst clips still show noise-specific oscillator failures: audible noise
  waves are sometimes predicted as pitched waves, and large `detune_amount` errors remain
  when the target detuned wave is noise.
- For a noise oscillator, detune is not a meaningful pitched offset in the same way it is
  for sine, triangle, saw, or square. Training hard on noise detune can therefore inject
  label noise into the oscillator objective.

The next model series should therefore be `v3.5`, focused on noise-aware detune loss
while keeping v3.4's target representation, architecture, and audibility-aware objective.
The preferred design is:

- Keep `main_detuned_mix` targets.
- Keep audibility-aware group balancing.
- Suppress or heavily reduce `detune_amount` loss when the detuned oscillator target wave
  is noise.
- Boost audible noise waveform classification so the model learns noise identity instead
  of chasing meaningless noise detune labels.

Acceptance criteria for `v3.5`:

- Fixed d8 1000-clip mean and median stay at least as good as v3.4's roughly `27.60` mean
  and `9.71` median.
- Worst clips involving target noise oscillators improve in weighted distance or show
  lower noise waveform error.
- High-detune pitched oscillator clips do not regress versus v3.4.

### Milestone I: Product Prototype - Windows Desktop App

Goal: turn NeuroWave from a research CLI into a usable single-platform desktop application
for producers. The first supported platform should be Windows because the current project
workflow, CUDA training setup, local file paths, and user environment are already Windows
first. macOS can come later after the app workflow is stable.

The first app should not train models inside the UI. It should run inference with a chosen
checkpoint, render the predicted synth patch, and let the user compare target audio against
the rendered prediction. Training remains a developer workflow until model quality and
packaging are stable.

Recommended product architecture:

- Desktop shell: Electron or Tauri.
- Frontend: local web UI with drag/drop, waveform cropper, transport controls, patch display,
  spectrogram display, and export actions.
- Backend: Python local inference service launched by the desktop shell.
- Model runtime: use the project `.venv` during development; package a dedicated Python
  runtime and checkpoint later.
- First platform: Windows x64.
- First app name: NeuroWave.
- First checkpoint: the current best v3 checkpoint, starting with `v3.5_noise_detune_loss`
  unless a newer benchmark clearly replaces it.

Core app workflow:

1. User drags a WAV file into the app.
2. App decodes the audio and shows the waveform.
3. User crops a one-note region with draggable start/end handles.
4. App previews the cropped region.
5. User enters or confirms pitch/frequency. Later this can be assisted by pitch detection.
6. App sends the cropped audio and frequency to the Python inference backend.
7. Backend predicts a patch JSON using the selected PyTorch checkpoint.
8. Backend renders the predicted patch to WAV.
9. App shows:
   - original cropped waveform,
   - rendered predicted waveform,
   - target spectrogram,
   - predicted spectrogram,
   - predicted synth parameters,
   - model/checkpoint metadata.
10. User can play A/B audio, save the predicted WAV, save the predicted JSON, and copy/export
    the patch report.

Minimum product features:

- Drag/drop WAV import.
- Audio decoding and mono preview.
- Waveform cropper with zoom and draggable handles.
- Crop playback.
- Frequency input in Hz, plus note-name helper such as A4 -> 440 Hz.
- Predict button with visible loading/progress state.
- Predicted patch JSON display.
- Rendered predicted WAV playback.
- Target/predicted spectrogram display.
- Export predicted WAV.
- Export predicted JSON.
- Save a local run folder under `playground/` or `runs/app/`.
- Error states for unsupported audio, missing checkpoint, missing frequency, invalid crop,
  backend crash, and CUDA/CPU runtime failure.

Product-quality backend requirements:

- Add a stable Python app API instead of wiring UI directly to scripts.
- Accept an input WAV path or uploaded/cached WAV bytes.
- Accept crop start/end seconds.
- Accept frequency in Hz.
- Convert stereo to mono deterministically.
- Save the cropped target WAV.
- Predict patch JSON with the selected checkpoint.
- Render the predicted patch WAV.
- Generate target and predicted spectrogram images or data arrays.
- Return a compact JSON response with all output paths and metadata.
- Keep generated app outputs out of git.

Suggested local backend API:

```text
POST /predict
  audio_path
  crop_start_seconds
  crop_end_seconds
  freq_hz
  model_path

returns:
  run_id
  cropped_target_wav
  predicted_patch_json
  predicted_wav
  target_spectrogram
  predicted_spectrogram
  model_metrics
  warnings
```

Suggested implementation order:

1. Extract reusable inference code from `scripts/playground_predict_wav.py` into a small
   `minisynth.app_inference` module.
2. Add crop-aware prediction/render helper tests.
3. Add a FastAPI or Flask local backend with `/health` and `/predict`.
4. Add a tiny local frontend prototype that can call `/health`.
5. Add drag/drop import and waveform preview.
6. Add crop selection and crop playback.
7. Add frequency input and note-name helper.
8. Connect the Predict action to `/predict`.
9. Display predicted JSON and rendered WAV playback.
10. Add target/predicted spectrogram display.
11. Add export buttons for WAV, JSON, and run folder.
12. Wrap the frontend/backend in a Windows desktop shell.
13. Add a first packaging build that can run on the development machine.
14. Add smoke tests for backend prediction and frontend app state.
15. Add product documentation and screenshots.

Acceptance criteria for the first desktop prototype:

- A producer can drag in a WAV, crop a one-note region, provide frequency, run prediction,
  and hear the rendered predicted audio without using the terminal.
- The app writes a predicted JSON and WAV into a deterministic local run folder.
- The app shows both target and predicted spectrograms.
- The app handles missing or invalid inputs with clear UI errors.
- The app can run with CPU fallback, even if CUDA is preferred when available.
- The prototype uses the existing model checkpoint format and does not require retraining.

### Milestone J: Product Website

Goal: publish a website for NeuroWave that explains the product, shows examples, and gives
users a place to download or request the desktop app.

The website should be a simple product site first, not an online inference product. Browser
inference can wait until model size, packaging, and rights around user audio are understood.

Recommended website architecture:

- Framework: Next.js or a simple static site.
- Hosting target: Vercel or another static-friendly host.
- First content:
  - product name and one-sentence promise,
  - short demo video or audio A/B examples,
  - screenshots of drag/drop, crop, prediction, and comparison views,
  - explanation that the first app is Windows desktop,
  - model limitations,
  - download/waitlist/contact section.

Website pages:

- Home: product value, screenshots, short demo, download/waitlist.
- How It Works: audio clip -> crop -> model predicts synth params -> render and compare.
- Examples: target clips, predicted clips, spectrogram comparisons, model version labels.
- Download: Windows build, install notes, GPU/CPU notes, known limitations.
- Changelog: model/app versions and benchmark highlights.

Website acceptance criteria:

- The first viewport clearly communicates NeuroWave as an audio-to-synth-parameter app.
- The website includes at least one real A/B example.
- It includes a Windows download or waitlist/contact flow.
- It documents that results depend on clean one-note clips and correct pitch/frequency.
- It links to model/app version notes so users understand limitations.

### Milestone K: Product Hardening

Goal: make the app reliable enough for repeated use outside the developer terminal.

Tasks:

- Add app logging.
- Add crash/error reports saved locally.
- Add model/checkpoint selection UI.
- Add CPU/GPU runtime indicator.
- Add app settings for default output folder and default model.
- Add recent files/runs list.
- Add run comparison history.
- Add installer packaging.
- Add signed Windows build if distribution requires it.
- Add automated smoke test that imports an example WAV and verifies prediction outputs.
- Add a simple release checklist.

## Important Technical Decisions

### Keep The Synth Small At First

The first AI goal is not to recreate every possible sound. It is to prove the loop:

```text
audio -> features -> parameters -> synth -> audio
```

A smaller synth with clean parameters is better than a huge synth with unclear controls.

### Prefer Continuous Parameters

Continuous parameters are easier for ML to learn than hard categories. For example:

- Waveform mixture is better than a single waveform enum.
- Filter mode can eventually be a soft blend, though an enum is acceptable early.
- Enabled/disabled modules can often be represented by amount values.

### Use A Fixed Parameter Vector

The ML system should start with a fixed-length vector. Avoid dynamic patch graphs until the basic inverse problem works.

### Separate Pitch From Timbre

For synthetic training, use the exact patch pitch as known context and focus the inverse model on timbre. The model should use pitch to interpret the mel spectrogram, because the same synth settings produce different spectra at different fundamentals, but it should not waste output capacity predicting `freq` when synthetic data already provides it exactly.

For real single-note clips, estimate pitch before model inference with a classical monophonic pitch estimator such as YIN or pYIN, and allow manual pitch input when automatic estimation fails. A separate learned pitch model should wait until classical pitch detection becomes a proven bottleneck.

Keep duration separate from pitch. Unlike `freq`, `length` affects ADSR interpretation and whether a sound behaves like a pluck, key, or pad, so it should remain visible to the model design until benchmark evidence shows a better representation.

### Save Everything

Every run should save:

- Input target path.
- Predicted patch JSON.
- Rendered WAV.
- Loss values.
- Model version.
- Synth version.
- Random seed.

This will make experiments reproducible.

## Risks

### Parameter Non-Uniqueness

Many different parameter configs can sound similar. The model may predict a different patch than the original but still sound correct.

This is not always a failure. Evaluation should include audio similarity, not only parameter accuracy.

### Synth Limitations

If the synth cannot produce a target sound, ML cannot solve it. The optimizer and loss should reveal these limits.

### Dataset Bias

If generated data is mostly harsh, quiet, or unrealistic, the model will learn that. Sampling rules matter.

### Real Audio Domain Gap

Models trained on clean synthetic audio may fail on real recordings. Real-clip matching should come after synthetic matching works.

## Immediate Next Step

The next implementation task should be the first product slice of Milestone I:

1. Extract the current external-WAV prediction flow into a reusable app inference module.
2. Add crop start/end support to that module.
3. Save a deterministic app run folder containing cropped target WAV, predicted JSON,
   predicted WAV, summary JSON, and spectrogram artifacts.
4. Add tests for crop validation, mono conversion, output paths, and prediction response shape.
5. Only after the backend helper is stable, add a local desktop/web UI around it.
