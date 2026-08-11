# G2 Complementary-Observable Analysis

## Decision

**COMPLEMENTARY_OBSERVABLE = INSUFFICIENT**

**EXPERIMENT_VALIDITY = NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN**

`INSUFFICIENT` means this current experiment did not establish usable information value. It does **not** prove that properly sampling-aware realized-variance observables are intrinsically incapable of helping.

**G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS.** The final representation is not frozen.

## Scientific observable contract

- `RV_SHORT`: annualized squared close-to-close log returns over the last `21` trading days.
- `RV_LONG`: the same statistic over `126` trading days.
- `RV_PERSISTENCE`: correlation of non-overlapping `5`-day realized-variance blocks separated by `4` blocks (`20` trading days), estimated from a `252`-day causal history.
- `ORACLE_TOTAL_VARIANCE`: `v0_slow + v0_fast`; this is an information upper bound, not a claimed market observation.

The 21/126-day pair was frozen because the four representative fast-factor e-folding times are about 34-68 trading days: 21 days is shorter, while 126 days is longer and begins to expose slow reversion. No alternative windows, lags, indicators, or feature combinations were searched.

For a CIR factor, the conditional trailing population variance under stationary reversibility is `theta + (v0-theta) * mean_j exp(-kappa*j/252)`. The persistence moment uses `Cov(v_t,v_(t+h)) = theta*sigma^2/(2*kappa) * exp(-kappa*h)` aggregated over the declared blocks, with the Gaussian squared-return variance in the denominator.

Primary references: [Heston (1993)](https://doi.org/10.1093/rfs/6.2.327), [Cox, Ingersoll and Ross (1985)](https://doi.org/10.2307/1911242), and [Andersen, Bollerslev, Diebold and Labys (2003)](https://doi.org/10.1111/1468-0262.00418).

One persistence scalar is only a weighted mixture of the two decay modes; it cannot identify both kappas by itself. The bounded D design follows the requested one-statistic limit and tests incremental information only. Bollerslev and Zhou's two-factor integrated-variance result motivates the decay mixture but does not make one sample autocorrelation an exact two-root estimator ([primary paper](https://doi.org/10.1016/S0304-4076(01)00141-5)).

## Synthetic construction and leakage boundary

Each case has an exact-CIR-marginal `252`-day variance history ending at the same `v0_slow/v0_fast` used by its option surface. Frozen seed `20260811` and case-derived seeds are recorded in `contract.json`. Returns use a disclosed daily Gaussian-copula leverage approximation. All windows end at valuation time; no future return enters an observable.

The historical path seed and shocks are never supplied to calibration. Calibration maps candidate parameters to population moments; therefore it does not replay the truth path or infer its innovations. The clean path retains intrinsic finite-history sampling variation. Added robustness noise is separate: option prices 0.5%/1.0% multiplicative; log-RV 0.05/0.10 standard deviation; persistence 0.05/0.10 absolute standard deviation. The oracle remains exact because it is explicitly an upper-bound diagnostic.

The synthetic experiment deliberately uses the same structural vector for return and option dynamics. In market language this is a `P=Q`/zero variance-risk-premium diagnostic assumption. Historical kappa/theta are physical-measure evidence, whereas the option engine is risk-neutral; no physical-to-risk-neutral bridge is established here, so this assumption cannot freeze a market input contract.

## A/B/C/D local identifiability

| Design | Median practical rank | Median smallest singular value | Median condition number |
|---|---:|---:|---:|
| A | 7.5/10 | 9.222e-03 | 6.556e+08 |
| B | 8.5/10 | 1.028e-02 | 1.110e+08 |
| C | 8.5/10 | 1.347e-02 | 1.189e+08 |
| D | 9.5/10 | 1.371e-01 | 1.641e+06 |

Rows are group-balanced and statistically scaled before SVD: the unchanged normalized-price equivalence scale, 0.10 log-RV, 0.10 persistence, and 1e-4 oracle total variance. Parameters remain scaled by full hard-bound widths exactly as in prior G2 work.

## Attempted clean global recovery and ambiguity

| Design | Median best parameter RMSE | Maximum best parameter RMSE | Near-equivalent | Materially displaced | Clusters | Median price RMSE | Median complementary RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1.214e-01 | 1.313e-01 | 40 | 39 | 39 | 4.708e-08 | n/a |
| B | 5.185e-02 | 9.353e-02 | 34 | 29 | 33 | 5.661e-08 | 2.646e-02 |
| C | 1.978e-01 | 3.757e-01 | 0 | 0 | 0 | no qualifying fit | no qualifying fit |
| D | 1.998e-01 | 2.213e-01 | 0 | 0 | 0 | no qualifying fit | no qualifying fit |

Zero qualifying C/D fits do **not** mean the ambiguity disappeared. They mean none of the reoptimized established option-equivalent regions simultaneously met the unchanged price threshold and the declared complementary-observable tolerance. The clean best complementary RMSE medians were 3.098 for C and 2.558 for D, both above the 1.0 screen.

The oracle B diagnostic lowered median best-fit parameter RMSE from 0.1214 to 0.0518, but retained 33 clusters and 29 materially displaced near-equivalent solutions; its cluster and material-solution reductions (15% and 26%) missed the 50% global thresholds. D lowered the median condition number from 6.556e8 to 1.641e6 (about 400x) and raised median practical rank from 7.5 to 9.5, yet median best-fit parameter RMSE worsened to 0.1998. This is local information gain without global recovery.

### Truth-fit validity diagnostic

| Case | Design | True-parameter complementary RMSE | Truth passes <=1 screen |
|---|---|---:|---|
| case_1 | C | 3.369 | False |
| case_1 | D | 2.890 | False |
| case_2 | C | 2.661 | False |
| case_2 | D | 2.333 | False |
| case_3 | C | 2.994 | False |
| case_3 | D | 2.561 | False |
| case_4 | C | 10.468 | False |
| case_4 | D | 8.658 | False |

The true vector fails the complementary screen in every C/D case because one finite 252-day path is compared with population moments using fixed 0.10 scales. Therefore C/D zero-fit and clustering rows are **invalid for global ambiguity inference**. They are retained as evidence that this observation-model/scaling contract is not fit for the decision, not as evidence that the underlying observables cannot help.

### Slow/fast, theta, and kappa errors

Median absolute full-range-scaled errors of the target-blind best-fit solution:

| Design | v0_slow | v0_fast | theta_slow | theta_fast | kappa_slow | kappa_fast |
|---|---:|---:|---:|---:|---:|---:|
| A | 4.336e-02 | 5.152e-02 | 2.517e-01 | 6.160e-02 | 2.077e-01 | 1.646e-02 |
| B | 5.524e-02 | 6.570e-02 | 4.202e-02 | 5.688e-02 | 1.069e-01 | 3.744e-02 |
| C | 1.512e-01 | 1.832e-01 | 1.193e-01 | 9.622e-02 | 2.254e-01 | 3.189e-01 |
| D | 8.533e-02 | 1.025e-01 | 9.506e-02 | 5.540e-02 | 2.818e-01 | 1.665e-01 |

## Noise robustness

The comparison below uses the identical `case_1;case_3` population (`n=2`) at every clean/noisy point. The four-case clean recovery and ambiguity evidence above remains the decision basis and is not replaced by this matched sensitivity view.

| Design | Option noise | Matched cases (n) | RV log-noise SD | Persistence noise SD | Median best parameter RMSE | Max best parameter RMSE | Bound hits |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 0.0% | case_1;case_3 (2) | 0.00 | 0.00 | 1.214e-01 | 1.240e-01 | 22/40 |
| A | 0.5% | case_1;case_3 (2) | 0.05 | 0.05 | 3.513e-01 | 3.518e-01 | 20/20 |
| A | 1.0% | case_1;case_3 (2) | 0.10 | 0.10 | 4.030e-01 | 4.454e-01 | 20/20 |
| B | 0.0% | case_1;case_3 (2) | 0.00 | 0.00 | 5.185e-02 | 5.262e-02 | 8/24 |
| B | 0.5% | case_1;case_3 (2) | 0.05 | 0.05 | 3.664e-01 | 3.782e-01 | 8/8 |
| B | 1.0% | case_1;case_3 (2) | 0.10 | 0.10 | 4.061e-01 | 4.508e-01 | 10/10 |
| C | 0.0% | case_1;case_3 (2) | 0.00 | 0.00 | 2.797e-01 | 3.757e-01 | 8/24 |
| C | 0.5% | case_1;case_3 (2) | 0.05 | 0.05 | 3.615e-01 | 3.684e-01 | 8/8 |
| C | 1.0% | case_1;case_3 (2) | 0.10 | 0.10 | 4.032e-01 | 4.454e-01 | 10/10 |
| D | 0.0% | case_1;case_3 (2) | 0.00 | 0.00 | 2.021e-01 | 2.213e-01 | 8/24 |
| D | 0.5% | case_1;case_3 (2) | 0.05 | 0.05 | 3.565e-01 | 3.585e-01 | 8/8 |
| D | 1.0% | case_1;case_3 (2) | 0.10 | 0.10 | 4.032e-01 | 4.454e-01 | 10/10 |

On the matched two-case population, at 0.5% and 1.0% option noise median best-fit parameter RMSE was about 0.36-0.37 and 0.40-0.41 across all designs; every usable B/C/D 1.0% solution hit a declared bound. The strict clean price-equivalence threshold retained no noisy solutions, so noisy cluster counts remain unresolved rather than zero ambiguity.

Across the 180 scheduled B/C/D fits, 174 produced valid finite capped iterates and six failed with the pricer's declared degenerate-denominator guard. No clean B/C/D fit satisfied SciPy's convergence termination within `max_nfev=80`; valid capped iterates were retained under the same evidence rule as the established global analysis. This limits any uniqueness claim and reinforces the fail-closed verdict.

## Real-market feasibility

`MARKET_OBSERVABLE_CONTRACT = UNRESOLVED`.

The predeclared maximum lookback is `252` trading days. The checkout contains only the three canonical valuation-date CM archives, not a continuous history. The official-NSE UDiFF acquisition/hash framework is reusable, and `ClsPric` is the declared close, but corporate-action adjustment, exchange-holiday completeness, missing-day rules, and a replayable adjusted-close contract are not implemented. No new data was acquired because the bounded feasibility decision is already unresolved and bulk acquisition would not cure those missing policies.

All four securities remain separate in `market_feasibility.csv`: NTPC, CIPLA, INFY, and HDFCBANK. Every requested return window is causal by construction, but it is not market-admissible until adjustment and completeness rules pass.

## Decision rule and boundary

A market design is materially informative only if, versus A, it reduces clean clusters and materially displaced near-equivalent solutions by at least 50%, lowers median best-fit parameter RMSE by at least 25%, and retains a near-equivalent fit in at least three of four cases. This rule was frozen before calibration.

Design triggers: `{'C': False, 'D': False}`. Therefore `COMPLEMENTARY_OBSERVABLE = INSUFFICIENT`.

**Mentor-ready numerical conclusion:** the exact total-variance oracle improved point recovery but left most separated option-equivalent regions intact. The sampled 21/126-day RV plus one persistence design cannot be judged globally because its fixed scales reject the truth in 4/4 cases; its apparent local conditioning gain does not rescue that invalid observation contract. The current experiment is insufficient evidence, not a proof of intrinsic observable insufficiency.

This experiment does not change the ten-parameter target, impose priors, reparameterize the model, generate the final 10k dataset, or train ANN/PINN. It does not modify prior G2 or checkpoint artifacts. G2 remains `NOT_PASSED` regardless of local conditioning gains.

**Single recommended next action:** predeclare and run a new sampling-aware synthetic design using empirically justified finite-window likelihood/scales and multiple fixed path seeds before any market-data acquisition.

## Reproducibility

- Canonical command: `python -B scripts/run_g2_complementary_observable_analysis.py`.
- Source-defined pre-run contract SHA-256: `F4CF48F5A3EE5939A030497748C2D0ED62C6C861BDEA0234F961C688797D5CD4`; it is written before calibration as `predeclared_contract.json`.
- Seed: `20260811`; node count: `64`; optimizer `max_nfev=80`.
- Protected prior files: `138`; aggregate SHA-256 `AAE30A596D6E95496C97867EE33DF2F2ADAE7F6115036BA135F988BACA4B9A50`.
- Design A comes from the preserved global-ambiguity CSV; B-D use those recovered vectors as target-blind warm starts.
- B-D reoptimize all 40 established clean near-equivalent A solutions (39 clusters); noisy runs use the five lowest-price-RMSE prior A solutions per matched case/noise level. This schedule was frozen after a no-output runtime timeout and before any B-D result existed.

## Mentor-ready figures

1. Conditioning comparison.
2. Ten-parameter error comparison.
3. Global ambiguity/clustering comparison.
4. Slow/fast variance allocation.
5. Theta/kappa information.
6. Matched-case clean versus noisy recovery (`case_1;case_3`, `n=2`).

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `ambiguity_summary.csv` | `72D55748DAF25CE856D5A94392292A76498DECC5A062FAB6689C1C56611144EE` |
| `cases.csv` | `7E9D2E2EE3F20194A7EDCBC9EF09C7F7DABF03BCDDCF89CD857F8447DBF05E2F` |
| `contract.json` | `76A977C4E7FD06C977F9EF7CD1DE3E744AF51B0505D5ED8181EB41262EEEC0C5` |
| `decision.json` | `5DEC41B90831658C79417D6943F177CBF0AC384494C8B344207DF87BFB05EAB8` |
| `experiment_matrix.csv` | `C14C3E79F00825F1B07B49DF364734D95A3887BE42DF139204671305B3C9D3A2` |
| `figures/01_conditioning_comparison.png` | `188BA6A1DB8CA2E432A3937396812C18523E2A8D591FCA3387C9A4A92999A08A` |
| `figures/02_parameter_error_comparison.png` | `5CB45F2AE2D49EE4C8CA580005E1684E56133F9C0F5E51E28CBBFBBB0DF4D3E0` |
| `figures/03_global_ambiguity_comparison.png` | `4EFA575B2C6133F79B19D191B367DC26AD9CB94B4DB89A8FBC45308095C2C36A` |
| `figures/04_slow_fast_variance_allocation.png` | `8156B75091D2AAF1D828B231BFE4F0A1DD7E7130D4BA161488D48AFE2813F365` |
| `figures/05_theta_kappa_information.png` | `93AF573BAC3D707D8508FF8EFB117C06EEAA9C49FEB185C51AC865D4F20843B4` |
| `figures/06_clean_vs_noisy.png` | `DB6E2B6ABF9E62ECF1E2D551042C65801EB9CAA3C475AE915215B2736BCBCDBB` |
| `jacobian_summary.csv` | `37314B3DF847DAA6248F543E35188FE9AF37C404737AEE41F31E6E14B62CD612` |
| `market_feasibility.csv` | `928B2E69C70264E790FD4EDA544B04B9F9934A2AB0C0263A9EDD95582A595802` |
| `parameter_errors.csv` | `0B46754F94D843F76A96917CEB3A0E45B24C9A3876A057E5A072719718FE10D6` |
| `parameter_sensitivities.csv` | `ED3DAB4C34B9CC97DFA51438B58D9CCABA61AF3CDB48DAD06F4C62E20EE6FF05` |
| `path_observables.csv` | `7D53A44F7000A929C185CE9029DC7B9AB8CC38FDFB17DB6F8AE3B2D52023BC2E` |
| `predeclared_contract.json` | `F4CF48F5A3EE5939A030497748C2D0ED62C6C861BDEA0234F961C688797D5CD4` |
| `recovery_solutions.csv` | `95A843737FDDE59301E673D71F90FCA59A47B543A4E97C80B0D933BDD9E6640A` |
| `recovery_summary.csv` | `AA2CA7A265F77F84600B700C43395209C3FE6D1EDCD0DBFAC4882893A9F699D3` |
| `singular_values.csv` | `A7F872F893CA4F806EE9B7EB1920146DE1B559AAD7436DAD92802CCBAAF5F998` |
| `synthetic_return_history.csv` | `C9DDA64BEEC588BBA6467BC83199DAC8EE844361658717BE13FC05D27F23144F` |
| `truth_fit_diagnostics.csv` | `BA85873B8276DE489820FF162D4047BB6463C325755528649C05E5B96C2CF787` |
| `weakest_directions.csv` | `B82D2E9F5D7BD3BB298C359CD33DC76C5D292EF542F2152DD940EC21DC9C4934` |

```text
COMPLEMENTARY_OBSERVABLE = INSUFFICIENT
EXPERIMENT_VALIDITY = NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN
MARKET_OBSERVABLE_CONTRACT = UNRESOLVED
G2 = NOT_PASSED
FINAL_REPRESENTATION = NOT_FROZEN
FINAL_10K_DATASET = NOT_GENERATED
ANN_TRAINING = NOT_STARTED
PINN_TRAINING = NOT_STARTED
```
