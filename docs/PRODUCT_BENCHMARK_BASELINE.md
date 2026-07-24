# Product Benchmark v1: Objective Baseline

Date: 2026-07-24 UTC

Benchmark: `neurowave_product_benchmark_v1` (36 fixed NWSD-v1 benchmark cases)
Status: objective baseline and blind listening review complete.

| Checkpoint | Mean weighted distance | Median | Maximum | Failed renders |
| --- | ---: | ---: | ---: | ---: |
| `v3.4_audible_loss` | 24.010 | 12.913 | 219.169 | 0 |
| `v3.5_noise_detune_loss` | 33.267 | 14.239 | 262.160 | 0 |

| Category | v3.4 mean | v3.5 mean | v3.5 minus v3.4 |
| --- | ---: | ---: | ---: |
| Wave identity | 10.056 | 48.696 | +38.639 |
| Pitched detune | 18.430 | 15.248 | -3.182 |
| Audible noise | 60.243 | 62.199 | +1.957 |
| Quiet mix | 27.072 | 23.407 | -3.665 |
| Envelope extreme | 15.043 | 39.174 | +24.131 |
| Filter/resonance extreme | 13.215 | 10.879 | -2.336 |

Negative deltas favour v3.5. It improves pitched detune, quiet mixes, and filter/resonance
extremes, but does not improve the targeted audible-noise slice and has material waveform and
envelope regressions on this product benchmark. The benchmark therefore does not support
promoting v3.5 over v3.4 on objective product-benchmark evidence alone.

## Blind listening result

The balanced 12-case A/B review was unblinded after scoring. v3.4 received 6 overall
preferences, v3.5 received 3, and 3 cases tied. Mean timbre scores were `4.75` for v3.4 and
`4.67` for v3.5; both received a mean envelope score of `5.00`.

The review agrees with the objective result in the audible-noise, envelope, and waveform
identity slices: v3.5 did not improve the intended noise behavior, and its waveform/timbre
regressions were audible. The listener also identified recurring pitch-alignment and
brightness/harmonic mismatch issues. The review is deliberately small, so it does not justify
an automatic public-checkpoint rollback; it does make v3.4 the control checkpoint for the
next experiment.

## Next experiment decision

Train one loss-only ablation: keep v3.5's architecture, data, optimizer, random seed, and
noise-detune suppression, but remove its additional audible-noise waveform-classification
boost. This isolates the unproven component most plausibly associated with the v3.5 waveform
and timbre regressions. Evaluate the resulting checkpoint against both v3.4 and v3.5 on the
unchanged 2,000-clip NWSD-v1 benchmark and this product benchmark before any promotion.

The local ignored reports and completed listening-review evidence are retained in
`runs/nwsd_v1/evaluation/archive/product_benchmark_v1_v3_4_v3_5_20260724/`. Regenerable
target/prediction WAV and copied patch artifacts were intentionally pruned after review.
