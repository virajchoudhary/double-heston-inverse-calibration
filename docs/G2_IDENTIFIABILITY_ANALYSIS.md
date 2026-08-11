# G2 Carry and Reduced-Grid Identifiability Analysis

## Decision

**G2 = NOT_PASSED.**

The market-supported central-5 geometry was not changed. This analysis conditions the inverse problem on maturity-aligned discount factors and normalized forwards, then tests only the canonical ten Double Heston targets. A low repricing error is not treated as parameter recovery.

## Predeclared carry contract

- Contract: `discount_forward_per_maturity_v1`.
- Input order: `[D_near, F_near/S, D_middle, F_middle/S, T_near, T_middle]`, then option-major (`call`, `put`), expiry-major (`near`, `middle`), moneyness ascending.
- Candidate dimension: **26** = 4 carry + 2 maturity + 20 normalized-price coordinates.
- Synthetic experiment term structure: `r=(0.0600, 0.0625)`, `q=(0.0200, 0.0225)`; the resulting `D` and `F/S` coordinates are known inputs, not fitted targets.
- Compatibility: the canonical scalar-carry engine is called separately per expiry after exact conversion from each `(D,F/S)` pair. The reviewed synthetic generator for this contract does not yet exist.
- Market limitation: official NSE spot/futures can provide `F/S`, but not `D=exp(-rT)`. The checked-in Stage A contract has no verified external short-rate/discount source or selected futures price field.

Alternatives were not silently adopted: explicit `(r_i,q_i)` contains equivalent information but encourages an unsupported interpretation that futures identify `q_i`; a carry-removing forward normalization changes `K/S` to `K/F` and therefore requires a new market-support audit. The current scalar `(r,q)` surface API was rejected because it would impose flat carry across two expiries.

## Experiment design

- Deterministic maximin sample: 8 valid vectors, balanced across reviewed interior and wide-valid evidence; all three observed near/middle DTE profiles were tested.
- Canonical production-pricer quadrature: `64` Gauss-Laguerre nodes for the full experiment.
- Jacobian: central finite differences of spot-normalized prices; each parameter column is scaled by its full hard-bound width.
- Practical rank threshold: singular value greater than `1e-06` times the largest singular value.
- Conditioning warning threshold: `1e+08`.
- Recovery: three target-blind deterministic starts, constrained latent transformation, and clean/0.5%/1.0% independent multiplicative price noise.
- Parameter recovery requires aggregate range-scaled RMSE <= 0.05 and maximum range-scaled error <= 0.15; optimizer convergence alone is insufficient.

## Jacobian results

Central-5 numerical rank 10 frequency: `100.0%`; practical rank 10 frequency: `0.0%`.
Smallest singular value: median `4.293e-09`, range `6.789e-12` to `9.636e-08`.
Condition number: median `5.107e+07`, 90th percentile `1.335e+10`, maximum `1.717e+10`.

Weakest median scaled-sensitivity parameters: `sigma_slow` (2.180e-03), `kappa_slow` (2.949e-03), `sigma_fast` (3.607e-03), `rho_slow` (3.700e-03), `rho_fast` (4.305e-03).
Dominant median absolute loadings in the weakest right-singular direction: `kappa_slow` (0.668), `theta_slow` (0.347), `theta_fast` (0.190), `v0_fast` (0.040), `kappa_fast` (0.033).

## Representation comparison

| Representation | Observables | Practical rank 10 | Median condition number | Market status |
|---|---:|---:|---:|---|
| `central5_calls_puts` | 20 | 0.0% | 5.107e+07 | PROPOSED_MARKET_GEOMETRY |
| `central3_calls_puts` | 12 | 0.0% | 1.201e+12 | PASS_BUT_DOMINATED |
| `central7_calls_puts` | 28 | 8.3% | 1.674e+07 | EVIDENCE_COMPARATOR_ONLY_STAGE_A_SUPPORT_FAIL |
| `central5_call_only` | 10 | 0.0% | 5.107e+07 | DIAGNOSTIC_PARITY_COMPARATOR_ONLY |

Calls and puts are parity-related conditional on known carry, so duplicating them does not create twenty independent stochastic-volatility observations. The call-only comparator quantifies this directly; both option types remain in the market representation for observed-data robustness, not as a claim of twenty independent equations.

## Deterministic multi-start recovery

| Noise | Cases | Optimizer success | Parameter-recovery success | Median best price RMSE | Median best parameter RMSE | Bound-hit starts | Median start variability |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0% | 4 | 6/12 | 0/12 | 2.515e-08 | 7.444e-02 | 5/12 | 1.368e-01 |
| 0.5% | 4 | 6/12 | 0/12 | 3.547e-04 | 3.312e-01 | 10/12 | 1.626e-01 |
| 1.0% | 4 | 7/12 | 0/12 | 6.268e-04 | 3.048e-01 | 10/12 | 9.179e-02 |

Median best-start absolute parameter error scaled by each hard-bound width:

| Parameter | Clean | 0.5% noise | 1.0% noise |
|---|---:|---:|---:|
| `kappa_slow` | 1.069e-01 | 4.360e-01 | 2.327e-01 |
| `theta_slow` | 1.292e-01 | 3.845e-01 | 1.297e-01 |
| `sigma_slow` | 2.672e-02 | 6.452e-02 | 1.455e-01 |
| `rho_slow` | 2.318e-02 | 2.983e-01 | 3.978e-01 |
| `v0_slow` | 3.114e-02 | 5.305e-02 | 1.736e-01 |
| `kappa_fast` | 1.721e-02 | 3.806e-01 | 3.945e-01 |
| `theta_fast` | 8.878e-02 | 1.921e-01 | 2.883e-01 |
| `sigma_fast` | 1.420e-02 | 3.485e-01 | 3.345e-01 |
| `rho_fast` | 6.685e-03 | 2.032e-01 | 6.218e-02 |
| `v0_fast` | 3.980e-02 | 6.198e-02 | 2.243e-01 |

Per-parameter errors, every optimizer start, constraint validity, bound diagnostics, and true/recovered vectors are retained in the CSV evidence. Best-start rows are selected by repricing error, not by knowledge of the true parameters.

## Gate components

| Component | Pass |
|---|---|
| `market_carry_pass` | `False` |
| `jacobian_rank_pass` | `False` |
| `conditioning_pass` | `False` |
| `clean_recovery_pass` | `False` |
| `noise_0_5pct_recovery_pass` | `False` |
| `noise_1pct_recovery_pass` | `False` |

## Minimum remedy

1. Add and provenance-validate a tenor-aligned external short-rate/discount source and select the official NSE futures price field, producing `(D_i,F_i/S)` without pretending futures identify `r_i` and `q_i` separately.
2. Treat the weak right-singular combinations reported above as the actual identifiability failure. Do not repair them by using the unsupported central-7 wings.
3. Reopen the market-supported information design: add independently supported maturities or complementary observables, reduce/reparameterize the ten targets, or introduce scientifically justified priors. Re-run the same rank and target-blind noisy-recovery gates afterward.

No final 10k dataset was generated. No ANN or PINN was trained.

## Reproducibility and preservation

The prior G2 analysis is replayed without writes before this experiment. Its four CSVs, three plots, and the eight canonical Stage A hashes are checked for preservation.

| New evidence artifact | SHA-256 |
|---|---|
| `representative_parameters.csv` | `9C95457CD856587EE97777320F71182851CC4514E777F1AC1BC963ECF04AA339` |
| `jacobian_summary.csv` | `B9184F5889A18B129494C0EAF13DDBC5E2A324B45DD3AE42F1870EED54B737C5` |
| `parameter_sensitivity.csv` | `760ED16999B340F21D51979C9D45BC383B237914CEDE2A76F38453F14117FFCC` |
| `near_null_directions.csv` | `C9426C19D26DEFFE7409BD21BD86264E5BF9CD9E20B0790559ECFBA8BFC70B6B` |
| `recovery_starts.csv` | `1D233F9D7F117CAD0099AE464A38A1EE66898854E5AA937420929887B30DC7FD` |
| `recovery_summary.csv` | `66217A886658BA993D9268763325EBF8D7BA856E970A4A48B0E89239D8B035EF` |
| `figures/jacobian_singular_values.png` | `0AF58A0474FB608818B3862A2E372766EA3FB6A833FE9B4C96B7095C12AF0625` |
| `figures/condition_number_distribution.png` | `5BA971E285A221EDBCBA4DC609B3ABA2E5BCDB53D5CBF31516DAEDF65E111D56` |
| `figures/parameter_sensitivity.png` | `185BDBA06C680F4A95DB4CB38CA10255A03E18A68379CF5222FB4CBC7B56C5FE` |
| `figures/clean_noisy_recovery.png` | `E9BA102967AA3960A33E7CA020060FE8F7028CA9D6F797487BDDDD94C443D570` |
| `figures/recovered_vs_true_clean.png` | `937E25553205B6EC733B0A18197C165E7BB35CC67FFE18DEC4E9629857A9DB9F` |

```text
CARRY_CONTRACT = discount_forward_per_maturity_v1
CANDIDATE_INPUT_DIMENSION = 26
G2 = NOT_PASSED
FINAL_SYNTHETIC_RESEARCH_DATA = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
```
