# Current Project Status

Status date: 13 August 2026

> Canonical current control and approval boundaries are maintained in [RESEARCH_CONTROL_AND_CURRENT_STATUS.md](RESEARCH_CONTROL_AND_CURRENT_STATUS.md). This file retains detailed background evidence.

## Approved research objective

Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

## Current completed milestone

The canonical Double Heston pricing/calibration foundation, independent pricing verification, reviewed synthetic-sampling foundation, ANN infrastructure, deterministic official-NSE Stage A candidate selection, and completed G2 diagnostics are checkpointed. PR #16 also completed the bounded NTPC optimizer-cap and three-date real-market information studies: optimizer-cap status is `OPTIMIZER_CAP_UNRESOLVED`, optimizer-only work is closed, and the three-date result is `MULTI_DATE_INSUFFICIENT`. Temporal information improved multi-start dispersion, but 15-Jul holdout RMSE deteriorated by `5.338%` and boundary-hit rate remained `1.0`. The final representation remains unfrozen and `G2 = NOT_PASSED`; the next action is the mentor decision recorded in [MENTOR_APPROVAL_BRIEF_G2_INFORMATION_DESIGN.md](MENTOR_APPROVAL_BRIEF_G2_INFORMATION_DESIGN.md).

The unavailable teammate engine is being replaced by an independently implemented canonical Double Heston engine. Equivalence to the unavailable source is not claimed.

| Component | Status |
|---|---|
| ANN project structure | Complete |
| Fixed ten-parameter contract | Complete |
| Canonical pricing engine | 36-case independent benchmark passed; freeze evidence created |
| Engine-focused benchmark set | 36 / 36 passed |
| Full automated suite | 238 passed at the complementary-observable milestone |
| Independent reference integrations | 36 / 36 reliable; zero warnings or failures |
| Production/reference agreement | All 64-node and 96-node cases passed |
| ANN research pricing adapter | Integrated |
| Clean controlled recovery | Completed; one exact best result, start sensitivity retained |
| One-percent noise recovery | Completed; parameter instability and boundary-near solutions observed |
| Genuine-engine pilot | 12 surfaces / 1,296 quotes |
| Prior provisional-bounds audit | 5,000 raw / 2,776 accepted / 2,224 rejected |
| Historical priced bounds-audit subset | 250 surfaces / 21,000 prices; all validity checks passed |
| Reviewed sampling audit | 19,000 candidates; interior 8,116 accepted, wide-valid 3,371 accepted |
| Normal synthetic core readiness | `CORE_DATASET_READY = true` under the existing candidate surface contract |
| Challenge stress readiness | `CHALLENGE_STRESS_READY = false`; retained stress cases remain separate |
| Stage A NSE screen | `COMPLETE`; 24 candidate stock surfaces across three dates |
| Primary Stage A source | Official NSE CM and F&O UDiFF bhavcopies |
| Candidate selection | `COMPLETE`; NTPC, CIPLA, INFY, and HDFCBANK |
| G2 market-supported geometry | `ESTABLISHED`; near + middle, central-five, calls + puts |
| Completed G2 diagnostics | Reduced-grid, third-expiry, multi-date A/B/C/D, independent CIR-path replication, global ambiguity, complementary observables |
| Global ten-parameter ambiguity | `ESTABLISHED`; 4/4 representative cases, 40 near-equivalent solutions, 39 clusters |
| Complementary-observable diagnostic | `INSUFFICIENT`; C/D truth outside fixed screen in 4/4 cases |
| Surface-representation G2 gate | `NOT_PASSED`; stable ten-parameter recovery not demonstrated |
| Current 108-input grid | `REJECTED_AS_FINAL_UNCHANGED_GRID` |
| Final G2 representation | `NOT_FROZEN`; no feature count frozen |
| Final 10,000-surface research dataset | Not generated |
| Development smoke test | Passing; remains `NOT_RESEARCH_DATA` |
| Full ANN research training | Not started |
| PINN development/comparison | Not started |
| Frozen real-market evaluation | Not started |

## Honest research boundary

The independent benchmark materially strengthens the pricing evidence, but agreement between two implementations is not proof of universal correctness. The historical 5,000-row uniform bounds audit did not approve that earlier sampling design: 44.48% of raw candidates were rejected, and among the 2,776 accepted vectors 32.6729% were near at least one declared boundary and 7.0605% were near a Feller boundary. Rejected vectors are reported separately and are not counted as boundary-near.

The reviewed four-distribution sampler separates normal training populations from deliberately difficult boundary-challenge and OOD populations. The normal clean core is ready under the current candidate surface contract, while retained challenge stress cases remain a separate non-ready evidence set. This does not establish externally valid parameter ranges, unique parameter recovery, ANN/PINN performance, or market generalization.

The 1% controlled noise experiment remains an important research result: comparatively reasonable price fit can coexist with severe deterioration in ten-parameter identification. This motivates the later ANN and physics-informed inverse comparisons rather than constituting evidence that either neural method is superior.

## Reviewed sampling follow-up

The reviewed sampler separates interior, wide-valid, boundary-challenge, and OOD populations. Interior acceptance is 81.16% and wide-valid acceptance is 67.42%; challenge rows are balanced across four labels and OOD rows lie outside normal `kappa_fast` support with no train/validation assignment.

Four deliberately difficult challenge surfaces produced tiny 64-node numerical-tolerance stress failures. They were retained as evidence, pass at 96 Gauss-Laguerre nodes, agree with the independent adaptive reference within the frozen comparison tolerance, and remain excluded from ordinary ANN training. Consequently normal core readiness and challenge stress readiness are tracked separately rather than forcing one global ready/not-ready label. See [REVIEWED_PARAMETER_SAMPLING.md](REVIEWED_PARAMETER_SAMPLING.md).

## Completed Stage A NSE market-support screen

The prepared synthetic plan still uses the candidate 108-input contract:

- 9 log-moneyness points;
- 6 candidate maturities: 7, 14, 30, 60, 90, and 180 days;
- calls and puts;
- 10 ordered Double Heston targets.

The mentor-updated v2 execution plan requires real-market support and inverse-identifiability evidence before final 10,000-surface generation. Stage A and G2 common-support supplied the market evidence. G2 identifiability diagnostics then showed that the supported geometry does not stably identify the canonical ten targets.

Stage A processed:

- Power candidates: NTPC and POWERGRID;
- Healthcare/Pharma candidates: SUNPHARMA and CIPLA;
- IT candidates: INFY and TCS;
- Financial/Banking candidates: ICICIBANK and HDFCBANK;
- NIFTY as a separate non-ranked reference;
- audit dates 01 July, 15 July, and 22 July 2026;
- one surface as one underlying-date containing near, mid, and far expiry slices together;
- price-usability checks separate from volume/open-interest activity;
- futures-implied carry inputs audited for availability without calculating or selecting final carry.

All eight stock candidates were present on all three dates. The result comprises 24 stock surfaces and 4,740 stock-option rows. Actual stock-option DTE patterns were `27/55/90`, `13/41/76`, and `6/34/69` by date. All 24 CM closes equaled the corresponding unique F&O `UndrlygPric`; all stock futures expiries aligned with stock-option expiries; and no selected stock row had an `XpryDt` / `FininstrmActlXpryDt` mismatch.

The current grid is not supportable unchanged as the final representation. The 180-day node was outside observed expiry support for all 24 surfaces, while the `-0.30` and `+0.30` log-moneyness wings were observed in only 2/72 and 7/72 stock-expiry slices. The central `-0.10` through `+0.10` nodes were observed in all 72 slices. This evidence rejects the final unchanged 108-grid but does not freeze a replacement. `G2 = NOT_PASSED`.

Official NSE is the primary Stage A source. Free NSE bhavcopy lacks historical bid/ask and quote-size fields, and Bloomberg was not used. The original three-date Power comparison remained unresolved; the separate five-Wednesday July extension selected NTPC at moderate confidence. The selected primaries are NTPC, CIPLA, INFY, and HDFCBANK. See [STAGE_A_CANDIDATE_SELECTION.md](STAGE_A_CANDIDATE_SELECTION.md).

## Completed G2 identifiability checkpoint

Common-support analysis established the near and middle listed expiries, five central log-moneyness nodes `[-0.10, -0.05, 0.00, +0.05, +0.10]`, and calls plus puts: 20 normalized option prices with maturity/carry conditioning treated separately.

The reduced-grid test had full algebraic rank but zero practical full-rank cases, approximately `5.107e7` median condition number, and `0/12` parameter-recovery passes at clean, 0.5%, and 1.0% noise. A third listed expiry improved the median smallest singular value by approximately `136.93×` and the median condition number by `65.47×`, but achieved only `12/24` practical full rank, `2/12` clean recovery, and `0/12` noisy recovery; it also failed the existing Stage A activity rule.

The multi-date A/B/C/D diagnostic showed that oracle multi-date information and exact CIR transition physics materially improve local conditioning. Its independent CIR-path replication was mixed only in the practical-rank behavior of latent-state C; the core recovery failure replicated. Maximum clean recovery remained `1/6`, and all 0.5% and 1.0% recovery results remained `0/6`.

```text
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
MULTI_DATE_DIAGNOSTIC = INSUFFICIENT
```

See [G2_IDENTIFIABILITY_CHECKPOINT.md](G2_IDENTIFIABILITY_CHECKPOINT.md) and the tracked [evidence manifest](evidence/G2_CHECKPOINT_MANIFEST.json).

## Completed global-ambiguity milestone

The clean primary diagnostic used four predeclared representative vectors, 20 deterministic starts per surface, the supported central-five calls-and-puts geometry, 64-node pricing, full-range parameter scaling, and a price-equivalence threshold frozen at normalized RMSE `2.5e-7`. It retained 40 near-equivalent clean solutions in 39 separated scaled-parameter clusters. Ambiguity was established in `4/4` cases; median price RMSE was `4.708e-8` while median range-scaled parameter RMSE was `0.1485`.

The optimizer-success-only subset retained 22 near-equivalent solutions, 21 materially displaced, across all four cases. The dominant repeated compensation was strongly negative `v0_slow/v0_fast`; repeated negative relationships also involved `theta_slow/theta_fast`, `sigma_slow/theta_fast`, and `rho_slow/theta_fast`. `kappa_slow/theta_slow` did not pass the global empirical screen.

Local/global displacement evidence is `CONSISTENT` in aggregate at median absolute cosine `0.535`, but not uniformly: two cases were consistent, one partially consistent, and one inconsistent. At both 0.5% and 1.0% noise every recovered vector hit at least one declared boundary and parameter error increased sharply. The mostly singleton clusters establish separated solution regions, not smooth basin volume.

```text
GLOBAL_AMBIGUITY = ESTABLISHED
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
```

See [G2_GLOBAL_AMBIGUITY_ANALYSIS.md](G2_GLOBAL_AMBIGUITY_ANALYSIS.md) and its [reproducibility manifest](evidence/G2_GLOBAL_AMBIGUITY_MANIFEST.json).

## Completed complementary-observable diagnostic

The frozen A/B/C/D comparison tested options only; options plus exact oracle total variance; causal 21-day and 126-day realized variance; and those realized-variance observables plus lag-4 autocorrelation of non-overlapping 5-day realized-variance blocks. Median practical rank increased from `7.5` for A to `9.5` for D, and the median condition number fell from `6.556e8` to `1.641e6`.

Those local gains did not establish global recovery. The oracle B design retained 33 clusters and 29 materially displaced near-equivalent solutions. C/D produced zero jointly qualifying fits because the true parameter vector itself failed the fixed complementary screen in all four cases; zero qualifying fits must therefore not be interpreted as ambiguity resolution. On the identical `case_1;case_3` noise panel, all designs had median parameter RMSE around `0.36` at 0.5% option noise and `0.40` at 1.0%.

```text
COMPLEMENTARY_OBSERVABLE = INSUFFICIENT
EXPERIMENT_VALIDITY = NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN
MARKET_OBSERVABLE_CONTRACT = UNRESOLVED
G2 = NOT_PASSED
```

See [G2_COMPLEMENTARY_OBSERVABLE_ANALYSIS.md](G2_COMPLEMENTARY_OBSERVABLE_ANALYSIS.md) and its [reproducibility manifest](evidence/G2_COMPLEMENTARY_OBSERVABLE_MANIFEST.json).

## Generation boundary

`CORE_DATASET_READY = true` means the reviewed normal parameter population and leakage/split logic are ready **under the existing candidate surface contract**. It does not authorize immediate final dataset generation under the mentor-updated pipeline.

The final 10,000-surface run remains blocked until the global parameter ambiguity is addressed, G2 approves a final representation, and the synthetic surface contract is updated and revalidated. No final research dataset, ANN research training, PINN derivation or training, or frozen real-market evaluation has occurred.

```text
STAGE_A_NSE_SCREEN = COMPLETE
STAGE_A_CANDIDATE_SELECTION = COMPLETE
POWER_SELECTION = NTPC
SELECTED_PRIMARY_SET = NTPC | CIPLA | INFY | HDFCBANK
BACKUP_SET = POWERGRID | SUNPHARMA | TCS | ICICIBANK
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED
MULTI_DATE_DIAGNOSTIC = INSUFFICIENT
GLOBAL_AMBIGUITY = ESTABLISHED
COMPLEMENTARY_OBSERVABLE = INSUFFICIENT
EXPERIMENT_VALIDITY = NOT_PASSED_TRUTH_OUTSIDE_COMPLEMENTARY_SCREEN
MARKET_OBSERVABLE_CONTRACT = UNRESOLVED
OPTIMIZER_CAP = OPTIMIZER_CAP_UNRESOLVED
OPTIMIZER_ONLY_WORK = CLOSED
NTPC_THREE_DATE_INFORMATION = MULTI_DATE_INSUFFICIENT
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
```
