# NTPC Double Heston multi-date calibration

## Why this was attempted

The corrected 160-vs-320 comparison was valid but returned `OPTIMIZER_CAP_UNRESOLVED`: cap incidence stayed 10/12 under both charts and dispersion persisted. Optimizer-only work is closed, so this predeclared three-date real-market test asks whether additional NTPC surfaces stabilize shared dynamics.

## Data and formulation

Official NSE UDiFF CM/F&O rows are used only for `2026-07-01`, `2026-07-15`, and `2026-07-22`, with the first two listed expiries, active actual strikes nearest `[-0.10,-0.05,0,+0.05,+0.10]`, inner targets for calibration and outer targets for holdout. `T=DTE/365` is reconstructed. The official RBI 91-day cut-offs are 5.2521% from Press Release 2026-2027/584 dated 1 July and 5.3324% from Press Release 2026-2027/672 dated 15 July; the latter is the latest available observation carried into 22 July. Both dated RBI HTML responses are locally preserved, hash-sealed, and field-validated before panel construction, so no future observation is used.

Eight structural parameters are shared; each date has its own `v0_slow,v0_fast`. Every date is priced by the canonical ten-vector, so this remains canonical Double Heston—not a new 14-parameter model. The joint loss uses raw price residuals divided by `sqrt(n_date)`; unweighted metrics are reported below. Twelve deterministic starts use the fixed 320 budget selected before results.

The 60-cell target inventory is preserved separately and missing cells are explicit rather than interpolated:

| date | selected cells | missing cells |
|---|---:|---:|
| 2026-07-01 | 11 | 9 |
| 2026-07-15 | 19 | 1 |
| 2026-07-22 | 18 | 2 |

| date | calibration rows | holdout rows | calibration RMSE | holdout RMSE | calibration IV RMSE | holdout IV RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07-01 | 8 | 3 | 0.588395978 | 0.287217379 | 0.0206440959 | 0.0307796171 |
| 2026-07-15 | 12 | 7 | 0.256976372 | 0.976300061 | 0.00854871758 | 0.0789182122 |
| 2026-07-22 | 12 | 6 | 0.255248455 | 0.165226828 | 0.0401934937 | 0.0812940961 |

## Best shared parameters

| parameter | value |
|---|---:|
| kappa_slow | 1.80808894 |
| theta_slow | 0.0855091908 |
| sigma_slow | 0.55607226 |
| rho_slow | 0.169221654 |
| kappa_fast | 3.65161805 |
| theta_fast | 0.0343337447 |
| sigma_fast | 0.500746822 |
| rho_fast | -0.934806935 |

| date | v0_slow | v0_fast | v0_total |
|---|---:|---:|---:|
| 2026-07-01 | 0.00500000012 | 0.0341289636 | 0.0391289637 |
| 2026-07-15 | 0.0411606329 | 0.0020000045 | 0.0431606374 |
| 2026-07-22 | 0.0362184883 | 0.00895510661 | 0.0451735949 |

Slow half-life is `139.926` days; fast half-life is `69.284` days. The primary direct-comparison canonical vector is the `2026-07-15` vector.

The reviewed single-date best had `kappa_slow=2.62553326` and `kappa_fast=11.7638631`, implying `96.3609`- and `21.5064`-day half-lives. The multi-date best shifts those to `kappa_slow=1.80808894` and `kappa_fast=3.65161805`, or `139.926` and `69.284` days.

Across near-equivalent starts, slow half-lives span `84.3329`–`4929.68` days for single-date and `139.807`–`721.13` days for multi-date; fast half-lives span `21.0832`–`84.3404` and `69.2463`–`721.109` days. Slow-kappa range/CV improve, but fast-kappa CV worsens and both multi-date half-life ranges remain broad. Timescale stability is therefore **mixed and insufficient**, not resolved.

Overall raw-price RMSE is `0.368440216` on calibration rows, `0.665367077` on holdout rows, and `0.487924083` over all selected rows. The best reported date-balanced objective is `0.398914551`.

## Stability and model comparison

| metric | reviewed single-date shared-8 | multi-date shared-8 |
|---|---:|---:|
| materially displaced | 11 | 7 |
| clusters | 7 | 3 |
| median separation | 0.399516908 | 0.324066116 |
| maximum separation | 0.627751647 | 0.481226608 |
| maximum distance from best | 0.54885039 | 0.481111526 |
| boundary-hit rate | 1 | 1 |
| cap rate | 0.833333 | 0.416667 |
| optimizer success rate | 0.166667 | 0.583333 |

Relative to the reviewed single-date shared-eight comparator, median separation improved `18.886%`, maximum separation `23.341%`, cluster count `57.143%`, and materially displaced count `36.364%`.

| shared parameter | minimum | maximum | range | coefficient of variation |
|---|---:|---:|---:|---:|
| kappa_slow | 0.350836629 | 1.80962923 | 1.4587926 | 0.346205075 |
| theta_slow | 0.0694170477 | 0.195023852 | 0.125606804 | 0.457301934 |
| sigma_slow | 0.369922948 | 0.556182099 | 0.18625915 | 0.133124604 |
| rho_slow | -0.949999978 | 0.16924171 | 1.11924169 | 1.16086466 |
| kappa_fast | 0.350846742 | 3.65360492 | 3.30275817 | 0.555213025 |
| theta_fast | 0.0343245953 | 0.199968223 | 0.165643628 | 0.754654075 |
| sigma_fast | 0.374588235 | 0.500813534 | 0.1262253 | 0.0934117168 |
| rho_fast | -0.934807162 | 0.356720196 | 1.29152736 | 2.48582568 |

For 15 July, reviewed single-date DH holdout RMSE was `0.92682472` and Standard Heston was `0.910569`. The multi-date 15 July holdout is `0.976300061`, a `5.338%` worsening that exceeds the predeclared 5% ceiling. Double Heston is not declared superior unless this evidence supports it.

The 12-start joint optimization consumed `969.869` seconds in total.

Final classification: **MULTI_DATE_INSUFFICIENT**.

The principal remaining limitation is that additional dates do not automatically turn local fit quality into unique recovery; shared timescales and allocations must be judged from the multi-start dispersion, cluster, boundary, and holdout evidence above.
