# G2 Information Remediation

## Independent baseline review

**Verdict: SHIP.** The previous negative identifiability result survived the fresh review.

The scaled Jacobian uses central differences of spot-normalized prices and full hard-bound parameter widths. Algebraic rank, the relative practical-rank threshold, condition numbers, target-blind recovery, constraint checks, bound diagnostics, and call/put parity redundancy are implemented consistently. Repricing and parameter recovery remain separate gates.

Independent step check on `interior_train_4151` at DTE `27|55`: rank `10`, smallest singular values `1.868e-10` and `1.867e-10` at relative steps `1e-4` and `1e-5`; the negative practical-rank conclusion is stable: `True`.

## Predeclared experiment matrix

The matrix was frozen before computing the three-expiry result. No wing search or combinatorial grid was performed.

| Representation | Listed expiries | Prices | Maturities | Carry | Candidate inputs | Market status |
|---|---:|---:|---:|---:|---:|---|
| `2exp_central5` | 2 | 20 | 2 | 4 | 26 | CURRENT_PROPOSED_MARKET_GEOMETRY |
| `3exp_central5` | 3 | 30 | 3 | 6 | 39 | SYNTHETIC_COMPARATOR_PENDING_FAR_MARKET_RULE |

Material-improvement trigger: at least `10x` gain in median smallest singular value or reduction in median condition number, with no practical-rank regression. The same eight parameter vectors, three actual date profiles, 64-node pricer, range scaling, thresholds, and deterministic protocol are used for both representations.

## Far-expiry market support

- `STRUCTURALLY_OBSERVED = YES`: central-5 bracketing uses the actual listed third expiry, with no maturity interpolation or extrapolation.
- `ACTIVELY_TRADED_UNDER_75PCT_RULE = NO`: overall mean `22.5%`, worst stock/date `0.0%`.
- `USABLE_UNDER_DECLARED_CLOSE_POLICY = YES`: worst close availability `100.0%`, worst settlement availability `100.0%`, largest bracket `0.044452`.
- `FAR_EXPIRY_MARKET_ADMISSION = NO`. The unchanged Stage A 75% activity rule is not redefined merely because the third expiry may improve conditioning.

| Stock | Date | DTE | Structural | Active | Close | Settlement | Price-policy usable | Activity-rule pass |
|---|---|---:|---:|---:|---:|---:|---|---|
| NTPC | 2026-07-01 | 90 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| NTPC | 2026-07-15 | 76 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| NTPC | 2026-07-22 | 69 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| CIPLA | 2026-07-01 | 90 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| CIPLA | 2026-07-15 | 76 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| CIPLA | 2026-07-22 | 69 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| INFY | 2026-07-01 | 90 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| INFY | 2026-07-15 | 76 | 100.0% | 70.0% | 100.0% | 100.0% | True | False |
| INFY | 2026-07-22 | 69 | 100.0% | 90.0% | 100.0% | 100.0% | True | True |
| HDFCBANK | 2026-07-01 | 90 | 100.0% | 0.0% | 100.0% | 100.0% | True | False |
| HDFCBANK | 2026-07-15 | 76 | 100.0% | 50.0% | 100.0% | 100.0% | True | False |
| HDFCBANK | 2026-07-22 | 69 | 100.0% | 60.0% | 100.0% | 100.0% | True | False |

## Discount-source provenance

This review is limited to INR sovereign-rate evidence for the Stage A valuation dates `2026-07-01`, `2026-07-15`, and `2026-07-22`. The selected near/middle option maturities are respectively `(27, 55)`, `(13, 41)`, and `(6, 34)` calendar days. No current rate was substituted for a historical observation.

### Official-source screen

| Official source | Series or instrument definition | Historical and tenor evidence | Assessment for stock-option discounting |
|---|---|---|---|
| [FBIL T-Bill Rate benchmark](https://www.fbil.org.in/#/benchmark/tbill), [historical table](https://www.fbil.org.in/), and [Version 3 methodology](https://www.fbil.org.in/uploads/T_Bill_Methodology_Document_Version_3_Dated_May_15_2023_a301f17397.pdf) | FBIL describes a money-market benchmark computed from secondary-market transactions in Government of India Treasury Bills and Cash Management Bills on NDS-OM. The threshold is three trades of at least INR 5 crore; executable orders may augment insufficient trades. CCIL is the calculation agent. FBIL publishes 14 nodes at 5:30 PM on Mumbai business days: 7D, 14D, and 1M through 12M. | The public date-filtered table returned all 14 nodes for each of the three requested dates during this run. FBIL's [January 2026 consultation](https://www.fbil.org.in/uploads/FBIL_Market_Consultation_T_Bill_and_Va_R_Jan_16_2026_2_4da45db358.pdf) says that, under the then-extant method, 14D, 1M, 2M, 3M, 6M, 9M, and 12M were transaction-supported key nodes, while 7D and the other intermediate nodes were interpolated or extrapolated. The current benchmark page still links Version 3 dated 2023; the consultation is not evidence that its proposed revision was adopted for July 2026. | Scientifically plausible sovereign short-rate proxy, but not frozen. It brackets every required maturity, yet 7D is itself model-derived under the extant method. Same-date 5:30 PM publication is later than the NSE equity-derivatives close, so using it for an earlier option snapshot may introduce look-ahead unless the valuation timestamp is explicitly after publication. A stable documented unauthenticated API or immutable dated download URL was not established; UI availability alone is insufficient for deterministic replay. |
| [CCIL NSS ZCYC parameters](https://www.ccilindia.com/en/zcyc-parameters) and [operational description](https://www.ccilindia.com/operational-aspects) | CCIL defines a Zero Coupon Sovereign Rupee Yield Curve using a Nelson-Siegel-Svensson spot-rate equation fitted to outright Central Government Security and T-Bill trades on NDS. | The official parameter page exposes dated curve parameters and the equation needed to compute a spot rate at an arbitrary maturity. CCIL says the curve is released daily at about 7:00 PM. A direct public query for the required July range did not produce the three required parameter records during this run; therefore no replayable dated source was validated. CCIL also says its workbook can be used only on the day of download and compares at most the previous 100 working days. | Best semantic match to a discount curve, and it avoids maturity-node interpolation in principle. It remains unvalidated for the required dates, is published later than FBIL, and lacks a verified dated raw artifact plus quote-to-discount convention for this run. |
| [RBI Government Securities primer](https://m.rbi.org.in/commonman/english/scripts/FAQs.aspx?Id=711) and RBI auction publications | RBI identifies T-Bills as zero-coupon short-term Government of India obligations issued at a discount and redeemed at par, describes Government securities as practically free of default risk, and specifies the money-market actual/365 convention. RBI auction outputs provide 91D, 182D, and 364D cut-off prices/yields. | Primary auction history exists, but it is auction-date rather than a daily valuation-date curve, and its original tenors do not align with 6-55 day option maturities. | Authoritative supporting provenance and convention evidence, not a sufficient standalone daily curve for this panel. Carrying forward the most recent auction yield would be an additional unvalidated stale-rate rule. |

### FBIL observations directly checked

The values below are the official FBIL table's displayed `Rate(%)` observations, not derived discount factors.

| Valuation date | 7D | 14D | 1M | 2M | Required maturity bracketing |
|---|---:|---:|---:|---:|---|
| 2026-07-01 | 4.86 | 4.94 | 5.13 | 5.20 | 27D: 14D-30D; 55D: 30D-60D |
| 2026-07-15 | 5.26 | 5.25 | 5.23 | 5.27 | 13D: 7D-14D; 41D: 30D-60D |
| 2026-07-22 | 5.23 | 5.22 | 5.21 | 5.20 | 6D: 0D-7D; 34D: 30D-60D |

The historical FBIL rows were individually queried for each exact date and carried a publication time of 5:30 PM. No values from `2026-07-08`, `2026-07-29`, or the present date were needed.

### Transparent maturity policy considered, but not adopted

If FBIL is later approved, a reproducible candidate policy is:

1. Use actual calendar days and an actual/365 year fraction; interpret benchmark labels as `7`, `14`, `30`, and `60` days.
2. Convert the benchmark yield to a node discount factor only after checking the operative FBIL quote convention. A candidate simple money-market conversion is `D(t) = 1 / (1 + y * t / 365)` with `y` expressed as a decimal.
3. Set `D(0) = 1` and interpolate linearly in `log(D)` between adjacent nodes. The 6D observation would use the segment from 0D to 7D; no maturity beyond 60D is needed for near/middle.
4. Never interpolate across valuation dates, carry a current rate backward, or silently replace a missing valuation-date record. A previous-business-day rule is a different temporal policy and must be predeclared and tested for look-ahead.

This policy is only a candidate. It does not cure the unresolved valuation timestamp, operative-methodology-version, raw-artifact, licensing, and replay issues.

### Conclusion

`DISCOUNT_SOURCE = UNRESOLVED`

Authoritative Indian primary sources exist and the FBIL table contains tenor-bracketing historical rates for all three dates, but a production discount source is not fully validated. Before selection, the pipeline must (1) freeze whether the curve timestamp is same-day after publication or prior-business-day relative to the NSE option snapshot, (2) archive a dated primary-source artifact with URL, retrieval timestamp, and hash for every valuation date, (3) verify the operative July 2026 methodology and quote-to-discount convention, and (4) demonstrate deterministic retrieval and replay under the source's use terms. Until all four gates pass, no discount factors should be generated from these rates and no current rate should be used for the historical surfaces.

## Identifiability comparison

| Representation | Algebraic rank 10 | Practical rank 10 | Median smallest singular value | Median condition number |
|---|---:|---:|---:|---:|
| `2exp_central5` | 100.0% | 0.0% | 4.293e-09 | 5.107e+07 |
| `3exp_central5` | 100.0% | 50.0% | 5.878e-07 | 7.801e+05 |

Median smallest-singular-value gain: `136.93x`; median condition-number reduction: `65.47x`; recovery trigger: `True`.

- `2exp_central5` weakest sensitivities: `sigma_slow` (2.180e-03), `kappa_slow` (2.949e-03), `sigma_fast` (3.607e-03), `rho_slow` (3.700e-03), `rho_fast` (4.305e-03).
- `2exp_central5` dominant weakest-direction loadings: `kappa_slow` (0.668), `theta_slow` (0.347), `theta_fast` (0.190), `v0_fast` (0.040), `kappa_fast` (0.033).
- `3exp_central5` weakest sensitivities: `sigma_slow` (4.830e-03), `rho_slow` (6.068e-03), `rho_fast` (6.603e-03), `sigma_fast` (6.886e-03), `kappa_slow` (7.692e-03).
- `3exp_central5` dominant weakest-direction loadings: `kappa_slow` (0.526), `theta_fast` (0.248), `theta_slow` (0.216), `v0_fast` (0.100), `v0_slow` (0.084).

## Target-blind recovery

| Representation | Noise | Optimizer success | Constraint valid | Recovery pass | Median price RMSE | Median parameter RMSE | Bound hits | Median start variation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2exp_central5` | 0.0% | 6/12 | 12/12 | 0/12 | 2.515e-08 | 7.444e-02 | 5/12 | 1.368e-01 |
| `2exp_central5` | 0.5% | 6/12 | 12/12 | 0/12 | 3.547e-04 | 3.312e-01 | 10/12 | 1.626e-01 |
| `2exp_central5` | 1.0% | 7/12 | 12/12 | 0/12 | 6.268e-04 | 3.048e-01 | 10/12 | 9.179e-02 |
| `3exp_central5` | 0.0% | 4/12 | 12/12 | 2/12 | 7.505e-08 | 5.339e-02 | 3/12 | 5.099e-02 |
| `3exp_central5` | 0.5% | 7/12 | 12/12 | 0/12 | 3.610e-04 | 3.195e-01 | 10/12 | 1.648e-01 |
| `3exp_central5` | 1.0% | 4/12 | 11/12 | 0/12 | 6.094e-04 | 3.498e-01 | 11/12 | 2.785e-03 |

Median best-start absolute parameter error scaled by each hard-bound width:

| Parameter | 2-exp clean | 3-exp clean | 2-exp 0.5% | 3-exp 0.5% | 2-exp 1.0% | 3-exp 1.0% |
|---|---:|---:|---:|---:|---:|---:|
| `kappa_slow` | 1.069e-01 | 5.621e-02 | 4.360e-01 | 4.357e-01 | 2.327e-01 | 2.287e-01 |
| `theta_slow` | 1.292e-01 | 4.902e-02 | 3.845e-01 | 4.352e-01 | 1.297e-01 | 2.691e-01 |
| `sigma_slow` | 2.672e-02 | 1.867e-02 | 6.452e-02 | 1.711e-01 | 1.455e-01 | 2.651e-01 |
| `rho_slow` | 2.318e-02 | 7.661e-03 | 2.983e-01 | 2.765e-01 | 3.978e-01 | 2.162e-01 |
| `v0_slow` | 3.114e-02 | 3.917e-02 | 5.305e-02 | 1.614e-01 | 1.736e-01 | 2.797e-01 |
| `kappa_fast` | 1.721e-02 | 1.487e-02 | 3.806e-01 | 1.633e-01 | 3.945e-01 | 4.589e-01 |
| `theta_fast` | 8.878e-02 | 5.556e-02 | 1.921e-01 | 1.562e-01 | 2.883e-01 | 2.274e-01 |
| `sigma_fast` | 1.420e-02 | 2.024e-02 | 3.485e-01 | 3.266e-01 | 3.345e-01 | 2.635e-01 |
| `rho_fast` | 6.685e-03 | 5.028e-03 | 2.032e-01 | 8.259e-02 | 6.218e-02 | 2.146e-01 |
| `v0_fast` | 3.980e-02 | 4.656e-02 | 6.198e-02 | 2.233e-01 | 2.243e-01 | 3.341e-01 |

## Decision

**G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS**

Three-expiry numerical pass: `False`; far-expiry market-quality pass: `False`; discount-source pass: `False`.

The final representation is not frozen unless both real-market policy and every numerical gate pass. No final 10k dataset was generated; ANN research training was not started; PINN work was not started.

## Phase 8 stop diagnosis and next action

The third listed expiry materially reduces local ill-conditioning but does not stabilize all ten targets: practical full rank is not universal, clean recovery remains below its predeclared frequency gate, and both noisy recovery gates fail. The dominant remaining weak combination continues to involve `kappa_slow`, `theta_slow`, and `theta_fast`; the result therefore points to missing slow-factor/time-evolution information rather than a need for unsupported strike wings.

Minimum defensible future choices are a complementary stock-specific variance observable, explicit weak-direction priors/regularization with prior-versus-data attribution, or a joint multi-date inverse observation model with an explicit state contract. None was implemented here.

**Single recommended next action:** predeclare a bounded joint multi-date, same-stock identifiability design using the already selected official-NSE option dates plus a reproducible variance-state observation, explicitly defining how `v0_slow` and `v0_fast` evolve so the canonical ten-parameter target is not reduced or silently redefined. Review that design with the mentor before any new data acquisition or calibration run.

## Reproducibility and artifacts

The eight canonical Stage A outputs, the prior common-support evidence, and the prior G2 identifiability evidence are hash-preserved before and after this run.

| New artifact | SHA-256 |
|---|---|
| `experiment_matrix.csv` | `2E454C1A12FC75285CEDAB90EBF46E9578A24B6935CD7A06DA0441EB33FC63D3` |
| `far_expiry_market_support.csv` | `F283E7B0036C06C588FECA35BCE6A02D4C53DBA3D67D7BA351EF82543A8EE126` |
| `jacobian_comparison.csv` | `10B4AB539FDDBE1F5BD214786C653ABE6B0DB2F5DDDFF95EFB2CE69E0F67BBD0` |
| `parameter_sensitivity_comparison.csv` | `C0AE84993CD7F5708221196D12CDF4FD66AFA5C7C3EBF8DF143FE9CD57DA1F27` |
| `weakest_directions_comparison.csv` | `89895F28871070B876A410D7C44B09F0CCB7CAF902442F4EA835278E4AD3CC6C` |
| `recovery_starts.csv` | `1C7B99CEF4DEA7891CFA3D3F81E0B17EF5923B6105C639856EF24CD0DCC900C2` |
| `recovery_summary.csv` | `E5E8069BEE23012EE30C420F9FD44BF6C043C4C7A7E66178887BC1F56EF41CE5` |
| `decision.json` | `EE3683910532FF41F43F14A736A5CE3B3C314E4EE95EEE5BABA05F4B74E4439F` |
| `figures/singular_values_comparison.png` | `AB3296EB592AC180156A113A07A9FA15116EEDCA177C0EFB1F3E120C530D23B9` |
| `figures/condition_number_comparison.png` | `90E233B37A3E2901ECDF26D358D3C1CAC1CA5C26C0EF7C254EB1326BC8FD9AFE` |
| `figures/practical_rank_comparison.png` | `18E9C0017D96F8F83AE3232C311B6B96FCA1FF6212D1237B72E4F909D68EC73E` |
| `figures/weakest_directions_comparison.png` | `0056E9AB6C16DEB7BBAA36302AFF5D447335221E52B80E65079D045CB60335C8` |
| `figures/recovery_comparison.png` | `EC5E6FE54943A761F13649C29A16A24BE5EE07D6C997FB2BE70BFFC6A19795EE` |
| `figures/far_expiry_market_support.png` | `51C3FEB0F08285B8F76FEA6CE90A239BEE2C5E33BCF01CDD3AE24AABB883283D` |
| `figures/mentor_summary.png` | `2622E8C2941BD865237122D98C09B70577F965773BD188B1C5972092626D6717` |

```text
INDEPENDENT_REVIEW = SHIP
PREVIOUS_NEGATIVE_IDENTIFIABILITY = SURVIVED_REVIEW
DISCOUNT_SOURCE = UNRESOLVED
FAR_EXPIRY_MARKET_ADMISSION = NOT_PASSED
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
FINAL_INPUT_DIMENSION = NOT_FROZEN
FINAL_SYNTHETIC_RESEARCH_DATA = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
```
