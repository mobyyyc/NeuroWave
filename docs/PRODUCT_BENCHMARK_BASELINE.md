# Product Benchmark v1: Objective Baseline

Date: 2026-07-24 UTC

Benchmark: `neurowave_product_benchmark_v1` (36 fixed NWSD-v1 benchmark cases)
Status: objective baseline complete; blind listening review pending.

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
promoting v3.5 over v3.4 on objective product-benchmark evidence alone. Do not change the
currently shipped checkpoint solely from this result: complete the blind listening review and
reconcile it with the broader 2,000-clip NWSD-v1 benchmark before making a release decision.

Local ignored artifacts are retained in:

- `runs/nwsd_v1/evaluation/product_benchmark/neurowave_product_benchmark_v1/v3.4_audible_loss/20260724T014654Z/`
- `runs/nwsd_v1/evaluation/product_benchmark/neurowave_product_benchmark_v1/v3.5_noise_detune_loss/20260724T014634Z/`
