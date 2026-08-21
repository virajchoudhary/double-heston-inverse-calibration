# G2 Global-Ambiguity Analysis

## Decision

**GLOBAL_AMBIGUITY = ESTABLISHED**

This bounded diagnostic is evidence about clean, target-blind multi-start recovery only. Noise is a stability probe and does not change the primary verdict.

## Mentor-ready numerical conclusions

- Collected `120` usable solutions: `80` clean and `40` noisy. Optimizer success was `49/120`; valid finite capped iterates were retained because excellent fit, not optimizer status, is the phenomenon under study.
- The clean screen retained `40/80` near-equivalent solutions across all four cases; `39` were materially displaced. Their median normalized price RMSE was `4.708e-08` while median range-scaled parameter RMSE was `1.485e-01` (range `2.891e-02` to `3.463e-01`).
- The result survives an optimizer-success-only robustness view: `22` near-equivalent successful solutions, `21` material, spanning all four cases and `22` separated clusters.
- The declared clustering produced `39` clusters; `38` are singletons and one has size two. Median nearest separation was `2.769e-01` in full-range coordinates. This establishes separated solution regions, but not the volume of smooth attraction basins.
- Across all clean starts, price RMSE and parameter RMSE had Spearman `0.799`; within the strict near-equivalent set, large parameter errors persisted at price errors as low as `2.574e-09`.
- Local/global evidence is `CONSISTENT` in aggregate (median absolute cosine `0.535`), but heterogeneous by case: two consistent, one partially consistent, and one inconsistent.

## Frozen contract

- Seed: `31415926`; representation: `central5_calls_puts`; production pricing: `64` nodes.
- Clean starts/case: `20`; noise starts/case: `10`; optimiser: TRF with `ftol=xtol=gtol=1e-10`, `diff_step=2e-05`, `max_nfev=120`.
- Near-price-equivalence threshold: `2.5e-07` normalized RMSE. It was rounded and frozen at ten times the prior clean median fit (`2.515e-8`) before this run; it was not tuned to the observed clusters.
- Material displacement: range-scaled parameter RMSE >= `0.05`.
- Clustering is deterministic complete linkage of clean, finite, constraint-valid, near-equivalent solutions at full-range distance <= `0.10`.

## Exact cases and true ten-vectors

- `case_1` (`interior_train_4151`, `2026-07-01`): `kappa_slow=0.72010013, theta_slow=0.062734392, sigma_slow=0.21197919, rho_slow=-0.18029788, v0_slow=0.073066243, kappa_fast=3.7034634, theta_fast=0.043319474, sigma_fast=0.2627966, rho_fast=-0.10360392, v0_fast=0.067885161`
- `case_2` (`interior_train_1450`, `2026-07-15`): `kappa_slow=1.3814304, theta_slow=0.11917047, sigma_slow=0.20286163, rho_slow=-0.66195713, v0_slow=0.10750457, kappa_fast=5.94028, theta_fast=0.073883749, sigma_fast=0.73114245, rho_fast=-0.061098393, v0_fast=0.077538064`
- `case_3` (`wide_valid_train_744`, `2026-07-22`): `kappa_slow=1.3454711, theta_slow=0.10402368, sigma_slow=0.19277381, rho_slow=-0.1243602, v0_slow=0.18306313, kappa_fast=6.9314441, theta_fast=0.077335982, sigma_fast=0.59055283, rho_fast=-0.2870653, v0_fast=0.12205414`
- `case_4` (`wide_valid_train_4264`, `2026-07-01`): `kappa_slow=2.2971879, theta_slow=0.19993119, sigma_slow=0.76110443, rho_slow=0.18275161, v0_slow=0.052356967, kappa_fast=7.4787633, theta_fast=0.064444036, sigma_fast=0.91159077, rho_fast=0.61430521, v0_fast=0.19585097`

## Case results

| Case | Profile | Near-equivalent | Clusters | Basin class | Boundary-associated | Ambiguous |
|---|---|---:|---:|---|---:|---|
| `case_1` | `2026-07-01` | 14 | 14 | `multiple_basin` | 3 | `True` |
| `case_2` | `2026-07-15` | 11 | 10 | `multiple_basin` | 0 | `True` |
| `case_3` | `2026-07-22` | 10 | 10 | `multiple_basin` | 5 | `True` |
| `case_4` | `2026-07-01` | 5 | 5 | `multiple_basin` | 1 | `True` |

## Cluster and noise numerical summaries

Full cluster sizes, center displacement, dispersion/diameter, separation, boundary association, and price/parameter RMSE ranges are in `cluster_summary.csv`. Most clusters are singleton solutions, so `multiple_basin` means separated solution regions under the declared cutoff; it does not estimate basin volume.

| Case | Noise | Usable | Near-equivalent | Basins | Basin class | Median price RMSE | Median parameter RMSE | Material | Bound hits |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `case_1` | 0.0% | 20 | 14 | 14 | `multiple_basin` | 1.161e-07 | 1.564e-01 | 14 | 8 |
| `case_1` | 0.5% | 10 | 0 | 0 | `single_or_unresolved` | 2.216e-04 | 2.608e-01 | 0 | 10 |
| `case_1` | 1.0% | 10 | 0 | 0 | `single_or_unresolved` | 5.996e-04 | 4.444e-01 | 0 | 10 |
| `case_2` | 0.0% | 20 | 11 | 10 | `multiple_basin` | 1.627e-07 | 2.088e-01 | 11 | 8 |
| `case_3` | 0.0% | 20 | 10 | 10 | `multiple_basin` | 3.348e-07 | 2.950e-01 | 10 | 14 |
| `case_3` | 0.5% | 10 | 0 | 0 | `single_or_unresolved` | 3.791e-04 | 3.489e-01 | 0 | 10 |
| `case_3` | 1.0% | 10 | 0 | 0 | `single_or_unresolved` | 5.220e-04 | 3.588e-01 | 0 | 10 |
| `case_4` | 0.0% | 20 | 5 | 5 | `multiple_basin` | 1.553e-06 | 2.613e-01 | 4 | 13 |

The noisy runs produced no solution below the strict clean-precision threshold, so noisy basin counts and noisy compensation pairs are **unresolved**, not evidence that ambiguity disappeared. Noise was evaluated only for the predeclared matched cases `case_1, case_3`. Within that matched population, median parameter RMSE changed from `0.203` clean to `0.332` at 0.5% and `0.371` at 1.0%; every noisy solution hit at least one declared boundary.

## Local-global alignment

Median absolute cosine for material solutions: `0.535`; classification: **CONSISTENT**. Case-level statuses are reported above. This is geometric alignment with the local weakest scaled-Jacobian direction, not a causal claim.

## Supported compensation pairs

Pairs require at least five clean near-equivalent solutions within a case and absolute Spearman correlation >= 0.5. They are descriptive co-movement only; they do not establish causal compensation. The full within-case screen is in `compensation_pairs.csv`.

- `v0_slow` / `v0_fast`: supported in `4/4` cases; median Spearman `-1.000` (median absolute `1.000`).
- `kappa_fast` / `v0_fast`: supported in `4/4` cases; median Spearman `-0.596` (median absolute `0.630`).
- `v0_slow` / `kappa_fast`: supported in `4/4` cases; median Spearman `0.596` (median absolute `0.612`).
- `rho_slow` / `theta_fast`: supported in `3/4` cases; median Spearman `-0.873` (median absolute `0.873`).
- `sigma_slow` / `theta_fast`: supported in `3/4` cases; median Spearman `-0.827` (median absolute `0.827`).
- `theta_fast` / `v0_fast`: supported in `3/4` cases; median Spearman `0.755` (median absolute `0.755`).
- `v0_slow` / `theta_fast`: supported in `3/4` cases; median Spearman `-0.755` (median absolute `0.755`).
- `theta_slow` / `theta_fast`: supported in `3/4` cases; median Spearman `-0.736` (median absolute `0.736`).
- `theta_fast` / `rho_fast`: supported in `3/4` cases; median Spearman `0.700` (median absolute `0.718`).
- `rho_slow` / `rho_fast`: supported in `3/4` cases; median Spearman `-0.700` (median absolute `0.700`).
- `theta_fast` / `sigma_fast`: supported in `3/4` cases; median Spearman `-0.700` (median absolute `0.700`).
- `rho_fast` / `v0_fast`: supported in `3/4` cases; median Spearman `-0.560` (median absolute `0.645`).
- `v0_slow` / `rho_fast`: supported in `3/4` cases; median Spearman `0.560` (median absolute `0.645`).
- `kappa_fast` / `theta_fast`: supported in `3/4` cases; median Spearman `-0.508` (median absolute `0.618`).

The explicitly suggested `theta_slow/theta_fast` relationship is supported negatively in three cases. `kappa_slow/theta_slow` did not meet the correlation screen despite both dominating several local weakest directions. The strongest repeated global relationship is the negative `v0_slow/v0_fast` variance-allocation trade-off in all four cases.

Noisy pair screens use the identical five-solution and absolute-Spearman thresholds; `noise_compensation_pairs.csv` records whether the dominant descriptive trade-offs persist or change.

## Six figures

1. Price RMSE versus range-scaled parameter RMSE.
2. True ten-vectors versus several near-equivalent vectors in scaled coordinates.
3. Strongest empirically supported compensation pair(s).
4. PCA cluster projection (no financial-coordinate interpretation).
5. Weakest local direction versus global-displacement cosine alignment.
6. Matched-case clean/0.5%/1.0% ambiguity stability summary.

## Ranked remedy categories

1. **Complementary observables** — add independent sensitivities that can separate the observed option-price-equivalent regions.
2. **Joint historical inference** — use time-series information to constrain persistence, long-run variance, and factor allocation; the completed multi-date option-only result shows that dates and exact CIR physics alone are not enough.
3. **Regularization / informative priors** — choose among weakly distinguished regions only when external scientific information justifies the prior; this stabilizes inference but does not make the prices identifying.
4. **Reparameterization** — expose combinations that the observations identify more directly while preserving the canonical scientific meaning; any target change requires a separate decision.
5. **Physics-informed inverse training** — use only after information content is addressed; a training architecture cannot manufacture uniqueness, and prior exact-CIR conditioning did not deliver stable recovery.
6. **Other: set-valued or uncertainty-set inference** — report observationally equivalent regions instead of a falsely precise point when uniqueness is not supported.

## Reproducibility and next action

Canonical optimization command: `python -B scripts/run_g2_global_ambiguity_analysis.py`. Replay status: `CANONICAL_RUN_COMPLETED_ONCE`.
CSV-only report/figure replay command: `python -B scripts/run_g2_global_ambiguity_analysis.py --render-only`. This path reads the preserved CSV/JSON artifacts and performs no optimization.
Exact latent-start schedule SHA-256: `7831622B03BFEE7AE3E4A5BFA5F458A7153F16198C20D8293139389B004F400E` using NumPy `default_rng`/`PCG64`; canonical runtime versions are preserved in the tracked manifest.
Protected pre-existing Stage A/G2 files: `115`; aggregate SHA-256 before and after: `727172F8E3DF71A056BB9434549461A6CD96CA82E94467040C91B4DD9C60A0C1`.
- Recommended next research action: mentor-review the clean cluster structure and its dominant compensation directions, then predeclare one complementary-observable or joint-historical experiment targeted at separating those regions.

## Gate boundary

G2 remains **NOT_PASSED**. The final representation remains unfrozen. No final dataset was generated and no ANN or PINN training was performed.

## Reproducibility artifacts

| Artifact | SHA-256 |
|---|---|
| `alignment.csv` | `4cb3e07e266171a3fff2b90a4690a0697cf1d9a77bc77facccff6633b7bfc1b3` |
| `all_solutions.csv` | `cab9fae2d1ed81445dfddb2fc38f8c05dcc69bf9260e87af82be505eb3fb10d8` |
| `cases.csv` | `7e9d2e2ee3f20194a7edcbc9ef09c7f7dabf03bcddcf89cd857f8447dbf05e2f` |
| `clean_near_equivalent.csv` | `fc1b5faa9b4b0077500a38d5eeb4426b163ad84228a79c945e57a1b6414377ba` |
| `cluster_summary.csv` | `3c154e1ab08a031714151587a409765fa45b735a116c154c24f473b68c1b27d1` |
| `clusters.csv` | `1469ed59df582daf55c8e806fd91d1faf46368018b3ceeea113480d4615ad39c` |
| `compensation_pairs.csv` | `43deb588915d710fcb95bb73dd3a093be640d8ee658ee9702875780187fb5936` |
| `contract.json` | `4ce5da63df2bb3bb4e4b2c06904763e1351008179f1a4ae6a8d43c814201a52f` |
| `decision.json` | `99ef940780cf502cb16323d6c915619cc635ffe5db5ed15e56ba8ab1965fe452` |
| `figures/01_price_rmse_vs_parameter_rmse.png` | `f560bff7d71003f63917a7445350d7a075c34e2a992bb939c188bb03aae30f02` |
| `figures/02_true_vs_alternative_scaled_vectors.png` | `6ebaf7ce69fe1532d41baa288b0777934d9ad2be7442a18a57cdb7a8189daae3` |
| `figures/03_compensation_pairs.png` | `1889770111283a2c7fc6176f4875790ce2fc3b1c4475112c8e09664bcd7bf66e` |
| `figures/04_cluster_projection_pca.png` | `8ac63c839c08023061a07091160aa1cc6bc94ec2c829a8899dcf78a8eb499c1e` |
| `figures/05_local_global_alignment.png` | `476144e401d1e6b9bf32896b6cc71ae1323d9ea244f587fb19959e5776f7659d` |
| `figures/06_clean_noise_ambiguity_summary.png` | `cf98778f42a7c79e7d9190d22ad2802041196f99abc844982e7e873f12516fc5` |
| `noise_compensation_pairs.csv` | `1cacd93c4e5a508a6eabc2b4b5abf6b3305df01542073f323c0da07c14c39d37` |
| `noise_summary.csv` | `2c120bfe40d64ed91ce1cc198c98fef5760abb5a84209dfbcdb008351f5fa6fe` |
| `summary.csv` | `3dfa27b1cf6834a91ff24d3a0bdd059b7489a0c7058b5de013738d23d23674ed` |
| `weakest_directions.csv` | `dbe422cbaec9c4a0fac0dcb828647eaa6276fe6bd8cdb3c815a7c819aff35add` |

```text
G2 = NOT_PASSED
FINAL_REPRESENTATION = UNFROZEN
FINAL_DATASET = NOT_GENERATED
ANN_TRAINING = NOT_STARTED
PINN_TRAINING = NOT_STARTED
```
