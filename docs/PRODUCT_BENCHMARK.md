# Product Benchmark v1

`datasets/product_benchmark_v1.json` is the fixed, versioned product-quality benchmark for
NeuroWave model comparisons. It selects 36 deterministic clips from the immutable
`nwsd_v1/benchmark` partition; it is not a training set, a development set, or a random
sample to regenerate between experiments.

## Coverage

Each category contains six unique cases:

- dominant pitched waveform identity;
- substantial pitched detune;
- audible noise;
- quiet oscillator mixes;
- envelope extremes; and
- filter or resonance extremes.

Each case records its release, partition, deterministic seed/index, exact pitch context,
category label, and scope limitation. The benchmark intentionally remains synthetic in v1.
Future external recordings must be added as a separately labelled domain-gap benchmark
version, never mixed with synthetic ground-truth comparisons.

## Validate

```powershell
& $Python scripts/validate_product_benchmark.py
```

## Required evaluation record

For every checkpoint comparison, save the benchmark ID, checkpoint path and ID, repository
revision, preprocessing settings, aggregate and per-category metrics, per-case metrics,
rendered target/prediction pairs, and ranked failure groups under
`runs/nwsd_v1/evaluation/`. Record a brief blind listening review for target/prediction pairs
on waveform/timbre similarity, envelope similarity, and overall usefulness (1–5 each).

Promote a candidate only when it avoids a material scientific-holdout regression, improves
its intended category, and does not lose the listening review. Change one model variable at a
time between comparisons.

## Run an evaluation

```powershell
& $Python scripts/evaluate_product_benchmark.py --model models/v3.5_noise_detune_loss.pt --device cuda --quiet
```

The evaluator creates a timestamped ignored run directory containing `report.json` and, for
each case, target/predicted WAV files plus target/predicted patch JSON. The report includes
aggregate metrics, category metrics, per-case metrics, ranked category failure groups, and
the highest-distance cases. Use a fresh empty `--output-dir` only when a predictable location
is needed.

The first objective checkpoint comparison is recorded in
`docs/PRODUCT_BENCHMARK_BASELINE.md`. It is not a promotion decision until the required blind
listening review is complete.

## Prepare a blind A/B review

Use two completed evaluator reports to create a balanced package with two high-disagreement
cases per category:

```powershell
& $Python scripts/prepare_product_benchmark_review.py `
  --report-a <v3.4-report.json> --label-a v3.4_audible_loss `
  --report-b <v3.5-report.json> --label-b v3.5_noise_detune_loss `
  --output-dir runs/nwsd_v1/evaluation/product_benchmark/neurowave_product_benchmark_v1/blind_review_v3_4_vs_v3_5
```

Listen to the target and each randomized A/B option in every case folder, then fill
`scores.csv`. Score timbre and envelope from 1–5, choose `a`, `b`, or `tie` for overall
usefulness, and add concise notes. Do not open `answer_key.json` until every row is scored.
