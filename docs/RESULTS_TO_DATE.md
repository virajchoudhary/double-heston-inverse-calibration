# Results to Date

Status date: 06 August 2026

## Verified engineering results

| Check | Fresh result |
|---|---|
| Full automated suite | 54 passed |
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

The detailed metrics, failed/stopped starts, and limitations are in [Double Heston validation results](DOUBLE_HESTON_VALIDATION_RESULTS.md).

## Interpretation

The clean controlled surface can be recovered to numerical precision from one deterministic start. Other clean starts stopped with low pricing error but different parameters. Under the fixed 1% noise realization, all starts stopped at the evaluation limit and produced similar pricing errors with unstable parameters; three were boundary-near. This is evidence of practical identifiability risk, not optimizer or model superiority.

## Results that do not yet exist

- ANN parameter-recovery results trained on the genuine canonical surfaces
- Broad 0%, 0.5%, 1%, and 2% multi-seed robustness results
- Confirmed empirical sampling bounds
- Chronological NIFTY EOD validation
- ANN versus PINN versus numerical calibration versus Standard Heston results

## Claims that must not be made

- Do not claim equivalence to the unavailable teammate source.
- Do not call provisional ranges original or teammate-confirmed bounds.
- Do not treat historical calibrated parameters as unique ANN truth.
- Do not treat the smoke test as financial evidence.
- Do not claim that synthetic validation proves real NIFTY performance.
- Do not describe the ANN or PINN as research-trained.
