# G2 Joint Multi-Date Identifiability Diagnostic

## Scope and preserved gate

This is a deterministic synthetic information-design experiment, not a real-market representation freeze. It does not redo Stage A, change prior G2 evidence, generate the final 10k dataset, train ANN/PINN, or change the established gate.

**G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS**

Controlled rates `(0.0600, 0.0625)` and dividend yields `(0.0200, 0.0225)` are used for each date's near/middle maturities. They are synthetic controls only and do not resolve the real-market discount-source contract.

Design A is the `2026-07-01` anchor-date control used for the joint-date comparison. Its statistics are therefore not expected to equal the established `5.107e7` condition-number median, which pooled all three independent single-date maturity profiles.

## Exact state contract

The canonical ten targets are the eight shared structural parameters plus `v_slow(t0)` and `v_fast(t0)`. Later states `v_slow(t1)`, `v_fast(t1)`, `v_slow(t2)`, and `v_fast(t2)` are date-specific. Designs C/D estimate them as nuisance variables; no design holds them equal to the anchor states.

Dates are `2026-07-01`, `2026-07-15`, `2026-07-22` with actual gaps `14` and `7` days. Near/middle maturity profiles are `(27, 55)`, `(13, 41)`, and `(6, 34)` days. Every surface uses central-five calls and puts.

Later states are sampled through the exact CIR noncentral-chi-square transition using fixed inverse-CDF uniforms. The sampled states are stochastic realizations, not conditional expectations.

## Predeclared A-D designs

| Design | Dates | Prices | Canonical targets | Nuisance states | State treatment |
|---|---:|---:|---:|---:|---|
| A | 1 | 20 | 10 | 0 | anchor date only |
| B | 3 | 60 | 10 | 0 | oracle known |
| C | 3 | 60 | 10 | 4 | latent independent |
| D | 3 | 60 | 10 | 4 | latent + exact CIR density |

## Nuisance-profiled identifiability method

A/B use the direct scaled price Jacobian. C removes the nuisance-state column space from the canonical-target Jacobian with an orthogonal SVD projection. D forms price information plus the positive-semidefinite outer product of exact CIR transition-log-density scores, scales the physics contribution by the squared predeclared 0.5%-reference normalized-price sigma, and profiles the four nuisance states through a Schur complement. All target columns use the previously validated full hard-bound range scaling and the relative `1e-6` practical-rank threshold.

## Critical comparison

| Design | Practical target rank 10 | Median condition number | Clean recovery | 0.5% | 1.0% |
|---|---:|---:|---:|---:|---:|
| A | 0.0% | 4.504e+07 | 0/6 | 0/6 | 0/6 |
| B | 100.0% | 7.261e+03 | 1/6 | 0/6 | 0/6 |
| C | 75.0% | 5.120e+04 | 0/6 | 0/6 | 0/6 |
| D | 100.0% | 6.279e+03 | 0/6 | 0/6 | 0/6 |

### Design A — single_date

Practical full target rank: `0.0%`; median smallest singular value: `5.994e-09`; median condition number: `4.504e+07`.

### Design B — multi_date_oracle_states

Practical full target rank: `100.0%`; median smallest singular value: `2.999e-05`; median condition number: `7.261e+03`.

### Design C — multi_date_latent_states

Practical full target rank: `75.0%`; median smallest singular value: `5.334e-06`; median condition number: `5.120e+04`.

### Design D — multi_date_latent_states_cir_physics

Practical full target rank: `100.0%`; median smallest singular value: `3.320e-05`; median condition number: `6.279e+03`.

## Recovery detail

| Design | Noise | Optimizer success | Canonical recovery pass | Median best price RMSE | Median best target RMSE | Median best nuisance RMSE | Bound hits | Start variability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.0% | 4/6 | 0/6 | 1.462e-06 | 1.389e-01 | N/A | 0/6 | 1.097e-01 |
| A | 0.5% | 1/6 | 0/6 | 2.590e-04 | 2.564e-01 | N/A | 5/6 | 2.349e-01 |
| A | 1.0% | 1/6 | 0/6 | 5.547e-04 | 3.578e-01 | N/A | 1/6 | 2.061e-01 |
| B | 0.0% | 2/6 | 1/6 | 3.689e-06 | 5.609e-02 | N/A | 2/6 | 1.582e-01 |
| B | 0.5% | 0/6 | 0/6 | 2.929e-04 | 2.981e-01 | N/A | 3/6 | 1.146e-01 |
| B | 1.0% | 1/6 | 0/6 | 5.713e-04 | 3.579e-01 | N/A | 5/6 | 1.296e-01 |
| C | 0.0% | 0/6 | 0/6 | 4.633e-06 | 7.473e-02 | 1.233e-02 | 1/6 | 1.197e-01 |
| C | 0.5% | 0/6 | 0/6 | 2.981e-04 | 1.469e-01 | 9.787e-02 | 3/6 | 1.290e-01 |
| C | 1.0% | 0/6 | 0/6 | 5.445e-04 | 2.159e-01 | 8.926e-02 | 2/6 | 1.468e-01 |
| D | 0.0% | 0/6 | 0/6 | 4.022e-05 | 1.944e-01 | 1.233e-01 | 2/6 | 2.014e-01 |
| D | 0.5% | 0/6 | 0/6 | 2.963e-04 | 2.210e-01 | 6.899e-02 | 3/6 | 1.852e-01 |
| D | 1.0% | 1/6 | 0/6 | 5.508e-04 | 2.320e-01 | 1.017e-01 | 3/6 | 1.516e-01 |

Nuisance-state errors for C/D are reported separately in `nuisance_state_recovery.csv`; they are never included in the canonical ten-target recovery gate. Recovery uses one representative target from each accepted distribution (two targets total), the same three deterministic starts, and an L-BFGS-B cap of `80` iterations; capped nonconvergence is retained as failure evidence. The eight-target local-identifiability sample is unchanged.

## Weakest remaining directions

- Design A: `kappa_slow` (0.673), `theta_slow` (0.336), `theta_fast` (0.208), `v0_fast` (0.060), `kappa_fast` (0.051).
- Design B: `theta_slow` (0.372), `kappa_slow` (0.343), `sigma_slow` (0.144), `theta_fast` (0.139), `sigma_fast` (0.117).
- Design C: `kappa_slow` (0.546), `theta_fast` (0.354), `theta_slow` (0.202), `v0_fast` (0.110), `v0_slow` (0.093).
- Design D: `theta_slow` (0.577), `kappa_slow` (0.262), `theta_fast` (0.246), `kappa_fast` (0.080), `rho_slow` (0.061).

## Decision

**MULTI_DATE_DIAGNOSTIC = INSUFFICIENT**

Critical interpretation: `CASE_1`. Oracle stop rule triggered: `False`. Optional `ORACLE_TOTAL_VARIANCE_DIAGNOSTIC` was not run; the bounded A-D experiment was sufficient for this decision.

`CASE_1` is assigned by the complete predeclared numerical gates: B passes the local practical-rank gate but fails stable target-blind recovery. This classification does not mean that multi-date information has no local conditioning value.

G2 is unchanged because this diagnostic does not satisfy or replace the real-market representation, far-expiry support, or discount-source provenance gates.

## Recommended next experiment

Run one mentor-reviewed synthetic replication with independently seeded CIR paths and the identical frozen A-D protocol; do not add features, priors, dates, or market proxies until the observed case replicates.

## Reproducibility and artifacts

Exact state paths: `8`. Protected prior Stage A/G2 artifacts were hashed before and after generation and remained byte-identical.

| Artifact | SHA-256 |
|---|---|
| `decision.json` | `B371DE5C5EAB23ADEAC685DB2D6920CCB3AED661562B178CD052B81BC92A00E7` |
| `experiment_designs.csv` | `97850DC00ABE68002327256CA2FA10F38875261DDC2B1CA57BEAC66120168F08` |
| `figures/condition_number_comparison.png` | `CC8DC8AA2896A46E84CE811A6F854A3B0A6C3224B26375DCB508AB37956FCF44` |
| `figures/mentor_summary.png` | `60310CFCD9AB976F4CA0BF400FA80B39F9A7D59CB91932120B7FB3CC5AB07D14` |
| `figures/nuisance_state_recovery.png` | `1027E07E6A96DBAF386C62A9F38E4E4DC21B6CF5A98939CC619A2AB8B462637E` |
| `figures/practical_rank_comparison.png` | `811F26D315543028AAAB736C472EB865B552F90BB7FA02C9075077AA52B9F2C7` |
| `figures/recovery_comparison.png` | `6D6C4B4BDA56F9776C62C4EE0D5DE13BAE165FF1E5A7D018F32C791F785B640B` |
| `figures/singular_values_comparison.png` | `D44996750C0AF6E6AF5D0EFCAD22CAB0594C9F99073F487DCA2C22D85E2E2F9C` |
| `figures/weakest_directions_comparison.png` | `9DE1CA48EE844B14174131B4A38ADA4D4D41BFE933D2D8B009CE65F92EB70BAE` |
| `identifiability_summary.csv` | `72FDDCEF1D9906CD54ABF5CC93F62C1EB56576C931170247FE3D076554C0AC02` |
| `nuisance_state_recovery.csv` | `C8E6A4C98D079B855A86A7B134D1EC83B14CD41A19875DB0F5E63BCF3F4D8A0F` |
| `parameter_error_summary.csv` | `1397287E040B8268D9CB8126DF02F3C7E1F9C28CEDD4D53CD54364F27866DC78` |
| `parameter_sensitivity.csv` | `CF7E1DDBCC2F9533B44462EF6929FA4E0D9B1915DF9AB3C3A4DD7B7461782902` |
| `recovery_starts.csv` | `0F08BA223CA1EDD5D9B386F5B2A359E425E12CE0D59595541D5DEC488761D86D` |
| `recovery_summary.csv` | `211B442F3E327DFB2B0185042DEF2E3C2DA6F5E98B376299C976C651CEAD0F2E` |
| `state_paths.csv` | `526FC95DE83F180D341EF426F0BDCE579485D73C915D84DADA19402259F0AD89` |
| `weakest_directions.csv` | `D5AD289A29006DA54F816D24E3A1BD9417F88A17CE21A1C3C18E8065754BE99C` |

MULTI_DATE_DIAGNOSTIC = INSUFFICIENT

G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
