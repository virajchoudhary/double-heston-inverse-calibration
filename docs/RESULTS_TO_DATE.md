# Results to Date

Status date: 11 August 2026

## Verified engineering results

| Check | Fresh result |
|---|---|
| Full automated suite | 219 passed at the global-ambiguity milestone |
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

## Verified Stage A NSE engineering results

| Check | Verified result |
|---|---:|
| Primary source | Official NSE CM and F&O UDiFF bhavcopies |
| Valuation dates | 01, 15, and 22 July 2026 |
| Candidate stock surfaces | 24: three for each of eight candidates |
| Selected stock-option rows | 4,740 |
| Candidate presence | All eight candidates present on all three dates |
| NIFTY | Derivatives present; reference-only and excluded from ranking |
| Stock futures rows | 72; every futures expiry matched an option expiry |
| Expiry-field mismatches | 0 in selected stock options and futures |
| CM/F&O spot checks | 24/24 CM closes exactly matched unique F&O `UndrlygPric` |
| Historical bid/ask and quote sizes | Not present in free NSE bhavcopy |
| Focused Stage A/provenance validation | 75 passed |
| Fresh full-suite validation | 219 passed at the global-ambiguity milestone |

The deterministic downloader/parser preserves official URLs, filenames, timestamps, archive sizes, ZIP and CSV SHA-256 hashes, ZIP integrity, member names, encoding, delimiter, and trading date. Raw and derived market-data files remain ignored and were not committed. See [Stage A NSE results](STAGE_A_NSE_RESULTS.md).

## Verified Stage A candidate-selection result

- Primaries: NTPC, CIPLA, INFY, and HDFCBANK.
- Backups: POWERGRID, SUNPHARMA, TCS, and ICICIBANK.
- The original three-date Power comparison was unresolved. The separately predeclared five-Wednesday July extension selected NTPC at moderate confidence.
- Bloomberg was not used.
- G2 common-support analysis across the four primaries is complete.
- `G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED`, `G2_FINAL_REPRESENTATION = NOT_FROZEN`, and `G2 = NOT_PASSED`.

See [Stage A candidate selection](STAGE_A_CANDIDATE_SELECTION.md).

## Verified Stage A market-support findings

- Actual stock-option DTE patterns were `27/55/90` on 01 July, `13/41/76` on 15 July, and `6/34/69` on 22 July.
- The central log-moneyness nodes from `-0.10` through `+0.10` were observed in all 72 stock-expiry slices.
- The extreme `-0.30` and `+0.30` nodes were observed in only 2/72 and 7/72 slices.
- The 30- and 60-day nodes were bracketed on all 24 surfaces; 180 days was outside observed expiry support on all 24.
- These observations reject the current 108-grid as the final unchanged representation but do not define a replacement.

## Verified G2 identifiability results

The market-supported geometry uses the near and middle revised/actual listed expiries, central log-moneyness nodes `[-0.10, -0.05, 0.00, +0.05, +0.10]`, and calls plus puts. It provides 20 normalized option-price observations; maturity and carry conditioning are considered separately.

| Diagnostic | Local information result | Recovery result | Gate consequence |
|---|---|---|---|
| Reduced-grid | Algebraic rank full; practical full rank `0/24`; median condition approximately `5.107e7` | Clean/0.5%/1.0% all `0/12` | Not passed |
| Third expiry | Smallest singular value `136.93×` better; condition `65.47×` better; practical full rank `12/24` | Clean `2/12`; noisy `0/12` | Not admitted; activity and recovery gates failed |
| Multi-date A/B/C/D | A/B/C/D median conditions `4.504e7`, `7.261e3`, `5.120e4`, `6.279e3` | Clean `0/6`, `1/6`, `0/6`, `0/6`; every noisy design `0/6` | `MULTI_DATE_DIAGNOSTIC = INSUFFICIENT` |
| Independent CIR-path replication | H1/H4 replicated; H2/H3 partially replicated because latent-state C practical rank was seed-sensitive | Maximum clean recovery remained `1/6`; every noisy design remained `0/6` | `REPLICATION = MIXED`; core recovery failure replicated |

The original and replication CIR path seeds were `20260811` and `27182818`. Multi-date information and exact CIR dynamics materially improve local conditioning, but stable global recovery of the canonical ten parameters remains poor.

```text
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
```

See [G2 identifiability checkpoint](G2_IDENTIFIABILITY_CHECKPOINT.md) and the [G2 evidence manifest](evidence/G2_CHECKPOINT_MANIFEST.json).

## Verified global-ambiguity result

The bounded global diagnostic collected 120 solutions across four predeclared representative parameter cases: 80 clean, 20 at 0.5% noise, and 20 at 1.0% noise. The clean equivalence threshold was normalized price RMSE `2.5e-7`; material displacement was full-range-scaled parameter RMSE `0.05`.

| Result | Verified value |
|---|---:|
| Clean near-equivalent solutions | 40 |
| Distinct scaled-parameter clusters | 39 |
| Representative cases with ambiguity | 4 / 4 |
| Median clean near-equivalent price RMSE | `4.708e-8` |
| Median clean near-equivalent parameter RMSE | `0.1485` |
| Clean near-equivalent parameter RMSE range | `0.0289` to `0.3463` |
| Optimizer-success-only near-equivalent solutions | 22; 21 materially displaced |
| Aggregate local/global absolute cosine | `0.535`; `CONSISTENT` |
| 0.5% median price / parameter RMSE | `2.998e-4` / `0.332` |
| 1.0% median price / parameter RMSE | `5.613e-4` / `0.371` |
| Noise boundary hits | 20 / 20 at each level |

The dominant repeated compensation was negative `v0_slow/v0_fast` in all four cases. Negative `theta_slow/theta_fast`, `sigma_slow/theta_fast`, and `rho_slow/theta_fast` relationships repeated across cases. `kappa_slow/theta_slow` did not pass the empirical global screen. Local and global evidence is consistent in aggregate but heterogeneous: two cases were consistent, one partially consistent, and one inconsistent. Because 38 of 39 clusters were singleton solutions, the evidence establishes separated parameter regions rather than smooth basin volume.

```text
GLOBAL_AMBIGUITY = ESTABLISHED
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
```

See [G2 global-ambiguity analysis](G2_GLOBAL_AMBIGUITY_ANALYSIS.md) and the [global-ambiguity manifest](evidence/G2_GLOBAL_AMBIGUITY_MANIFEST.json).

## Verified complementary-observable result

The bounded A/B/C/D experiment retained the canonical ten-parameter target and tested only predeclared incremental observables.

| Design | Observables | Median practical rank | Median condition number | Four-case clean parameter RMSE |
|---|---|---:|---:|---:|
| A | Options only | 7.5 | `6.556e8` | `0.121398` |
| B | A + oracle `v0_slow + v0_fast` | 8.5 | `1.110e8` | `0.051845` |
| C | A + causal 21/126-day realized variance | 8.5 | `1.189e8` | `0.197754` |
| D | C + lag-4 autocorrelation of non-overlapping 5-day realized-variance blocks | 9.5 | `1.641e6` | `0.199805` |

Design B retained 34 near-equivalent solutions, 29 materially displaced solutions, and 33 clusters. C/D had zero jointly qualifying fits, but the truth itself failed the fixed complementary screen in 4/4 cases, so those zeros are invalid as ambiguity-resolution evidence. On the identical two-case noise panel, the A/B/C/D median parameter RMSEs were `0.3513/0.3664/0.3615/0.3565` at 0.5% noise and `0.4030/0.4061/0.4032/0.4032` at 1.0% noise.

```text
COMPLEMENTARY_OBSERVABLE = INSUFFICIENT
EXPERIMENT_VALIDITY = NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN
MARKET_OBSERVABLE_CONTRACT = UNRESOLVED
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
```

See [G2 complementary-observable analysis](G2_COMPLEMENTARY_OBSERVABLE_ANALYSIS.md) and the [complementary-observable manifest](evidence/G2_COMPLEMENTARY_OBSERVABLE_MANIFEST.json).

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
- A frozen replacement neural representation or a passed G2 gate
- Stable recovery of all ten canonical parameters under the market-supported geometry
- Bloomberg historical quote-quality evidence
- The final 10,000-surface research dataset

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
- Do not treat market-data availability as proof of parameter identifiability.
- Do not claim that the replacement representation is complete or that G2 has passed.

```text
STAGE_A_NSE_SCREEN = COMPLETE
STAGE_A_CANDIDATE_SELECTION = COMPLETE
POWER_SELECTION = NTPC
SELECTED_PRIMARY_SET = NTPC | CIPLA | INFY | HDFCBANK
BACKUP_SET = POWERGRID | SUNPHARMA | TCS | ICICIBANK
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
REPLACEMENT_REPRESENTATION = OPEN
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
MULTI_DATE_DIAGNOSTIC = INSUFFICIENT
GLOBAL_AMBIGUITY = ESTABLISHED
COMPLEMENTARY_OBSERVABLE = INSUFFICIENT
EXPERIMENT_VALIDITY = NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN
MARKET_OBSERVABLE_CONTRACT = UNRESOLVED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
```
