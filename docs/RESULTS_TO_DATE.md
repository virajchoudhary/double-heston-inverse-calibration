# Results to Date

Status date: 07 August 2026

## Verified engineering results

| Check | Fresh result |
|---|---|
| Full automated suite | 86 passed, including 2 readiness-contract tests |
| Engine-focused tests | 36 passed |
| Canonical fixture | 18 quotes; deterministic and reproducible |
| No-arbitrage bounds | Passed |
| Maximum parity error | `1.4210854715202004e-14` |
| 64-vs-96-node maximum difference | `4.291678123991005e-12` |
| Best clean price RMSE | `1.6125487393436679e-13` |
| Best clean parameter RMSE | `2.2178553561322553e-11` |
| Realized noise RMS | `0.010574835822177494` |
| Selected noisy price RMSE | `0.17631291031892615` |
| Selected noisy parameter RMSE | `0.43113598206350984` |
| Optimizer starts | 3 clean + 3 noisy; every start recorded |
| Boundary-near candidates | 3, all in noisy experiment |
| Genuine-engine pilot | 12 surfaces / 1,296 quote rows |
| ANN adapter | Calls the independent canonical engine |
| Full ANN research training | Not started |

## Independent benchmark and bounds audit

| Check | Fresh result |
|---|---:|
| Frozen benchmark cases | 36: 18 calls + 18 paired puts |
| 64-node RMSE / MAE | `5.458369984817452e-13` / `5.18369298103178e-13` |
| 96-node RMSE / MAE | `4.2228670813888515e-12` / `4.0641980521745795e-12` |
| Maximum absolute difference | `8.100187187665142e-13` (64); `5.6985527407960035e-12` (96) |
| Reference warnings / unreliable integrations | 0 / 0 |
| Benchmark no-arbitrage / parity failures | 0 / 0 |
| Prior raw audit candidates | 5,000 |
| Reviewed sampling candidates | 19,000 across four populations |
| Reviewed interior / wide accepted | 8,116 / 3,371 (`81.16%` / `67.42%`) |
| Reviewed challenge / OOD valid | 2,000 / 2,000; 500 per challenge label |
| Reviewed priced-surface failures | 4 retained challenge failures; no rows dropped |
| Historical priced bounds-audit surfaces | 250; 21,000 finite prices |
| Historical bounds-audit surface validity failures | 0 bounds, monotonicity, or convexity failures |
| Similar-surface/separated-parameter pairs | 17 |
| Freeze decision | `NEEDS_SAMPLER_CORRECTION` |

The detailed benchmark, bounds, controlled-calibration, and freeze evidence are in [Independent pricing benchmark](INDEPENDENT_PRICING_BENCHMARK.md), [Parameter-bounds audit](PARAMETER_BOUNDS_AUDIT.md), [Double Heston validation results](DOUBLE_HESTON_VALIDATION_RESULTS.md), and [Engine freeze](ENGINE_FREEZE.md).

## Interpretation

The clean controlled surface can be recovered to numerical precision from one deterministic start. Other clean starts stopped with low pricing error but different parameters. Under the fixed 1% noise realization, all starts stopped at the evaluation limit and produced similar pricing errors with unstable parameters; three were boundary-near. This is evidence of practical identifiability risk, not optimizer or model superiority.

## Results that do not yet exist

- ANN parameter-recovery results trained on the genuine canonical surfaces
- Broad 0%, 0.5%, 1%, and 2% multi-seed robustness results
- Financially approved empirical sampling bounds
- Chronological NIFTY EOD validation
- ANN versus PINN versus numerical calibration versus Standard Heston results
- A generated reviewed-core ANN dataset or any ANN training result

## Prepared reviewed-core pilot

The 10,000-surface normal-clean plan is prepared only; it has not generated a
dataset, trained an ANN or PINN, validated real NIFTY data, or changed the global `NEEDS_SAMPLER_CORRECTION`
decision. Its boundary challenge, OOD, and raw-noise populations remain
separate from the core.

## Claims that must not be made

- Do not claim equivalence to the unavailable teammate source.
- Do not call provisional ranges original or teammate-confirmed bounds.
- Do not treat historical calibrated parameters as unique ANN truth.
- Do not treat the smoke test as financial evidence.
- Do not claim that synthetic validation proves real NIFTY performance.
- Do not describe the ANN or PINN as research-trained.
