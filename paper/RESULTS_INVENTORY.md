# Central results inventory

This is the authoritative claim-to-source map for the paper synthesis layer. The
current synthesis base commit is
`72ad8e1aa845ec4c6f0fc61fc526df75438639bb` (`BASE`). External evidence is read only
by explicit Git refs:

- `NOISE=e58b36d1ff6cbf39115d46698ad0fb45b1ac1b8b`
  (`origin/codex/r2-noise-recovery`);
- `MODEL3=02c2a2cbc2498d5c4ce1e914e7c3d22693a55fc9`
  (`origin/research/model3-pde-protocol`).

The branches are not merged into this worktree. Tables live under
`paper/generated/tables/`; PDFs are generated under `paper/generated/figures/` and
byte-copied to `paper/figures/`, the LaTeX graphics path.
`paper/scripts/generate_results_assets.py` writes them from committed artifacts without
rerunning science.

## Evidence classification

| Item | Classification | Source artifact | Commit/ref |
|---|---|---|---|
| Canonical Double Heston implementation, independent benchmark | Canonical result | `outputs/double_heston_benchmark/benchmark_summary.json`; `docs/DOUBLE_HESTON_VALIDATION_RESULTS.md`; `docs/INDEPENDENT_PRICING_BENCHMARK.md` | `BASE` |
| Official-NSE Stage A screening and candidate selection | Development/provenance context | `docs/STAGE_A_NSE_RESULTS.md`; `docs/STAGE_A_CANDIDATE_SELECTION.md` | `BASE` |
| NTPC BS/Heston/DH pilot on 2026-07-15 | Development result; NO_CLEAR_WINNER | `docs/NTPC_SINGLE_STOCK_CALIBRATION.md`; `docs/evidence/NTPC_SINGLE_STOCK_PILOT_MANIFEST.json`; ignored derived CSVs listed by manifest hash | `BASE` |
| Pre-R2 G2 geometry, multi-date, global ambiguity, complementary observables | Superseded/historical motivation; practical non-identifiability retained | `docs/G2_IDENTIFIABILITY_CHECKPOINT.md`; `docs/G2_GLOBAL_AMBIGUITY_ANALYSIS.md`; `docs/evidence/G2_GLOBAL_AMBIGUITY_MANIFEST.json`; `docs/RESULTS_TO_DATE.md` | `BASE` |
| R2/R3 representation selection | Completed decision; canonical selection of R2 | `evidence/g2_r2_r3_20260822/final_decision.json`; `diagnostics_summary.json`; `manifest.json`; `docs/G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md` | `BASE` |
| Frozen R2 interface | Canonical result | `docs/R2_REPRESENTATION_CONTRACT.md`; `src/r2_representation/contract.py` | `BASE` |
| Final 10k clean synthetic dataset | Canonical dataset | `data/final_r2_clean_10000/manifest.json`; `integrity_report.json`; `replay_report.json`; surfaces SHA `148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6` | generation commit `b13caf0747d339c7a6bdf7a0400465bcfaad814f`, present in `BASE` history |
| Primary Traditional/ANN/Model2 comparison | Canonical completed result | `evidence/r2_primary_comparison_20260823/FINAL_EVALUATION_EVIDENCE_MANIFEST.json`, `synthetic_test_comparison.csv`, metric JSONs, journals | merge/base `BASE`; final-evidence generation after `285a7247eed5096756caec0d561f2f69be140d52` |
| Model2 local CPU seed-11 replication | Historical/non-primary replica | `training_run_manifest.json`; `P100_MODEL2_COHORT_MANIFEST.json` | `BASE` |
| Noise cohorts and strict 0% gates | Completed protocol subsets | `origin/codex/r2-noise-recovery:evidence/r2_noise_robustness/**`; protocol at `91b66b63af7fbda6ad425fe7beeddf045e6b99c0` | commits through `NOISE` |
| Full-population neural noise evaluations (0%-1%) | Completed neural-only result | `origin/codex/r2-noise-recovery:evidence/r2_noise_robustness/neural/all_neural_seed_headline.csv`; `neural/MANIFEST.json` | neural evaluation `4ffddc2b9170ba494f75cedb2f5740cb806558c9`; inventory ref `NOISE` |
| Traditional positive-noise subset | Active/pending | `origin/codex/r2-noise-recovery:docs/R2_NOISE_CLOUD_EXECUTION_HANDOFF.md`; interrupted 0.10% journal | preparation/handoff `NOISE`; execution target `b94447d71d8418a218e0e5f2a91ba807f2dcb687` |
| Genuine PDE-informed Model3 | Active/pending methodology only | `origin/research/model3-pde-protocol:docs/MODEL3_PDE_PROTOCOL.md`; `configs/model3_pde_protocol.yaml` | `MODEL3` |
| OOD/boundary robustness | Pending; no result | No completed OOD result artifact exists in audited current evidence | Not applicable |
| G8 real-market evaluation | Not started; no result | R2 contract marks five NTPC dates development-only and excluded from G8 | `BASE` |

## Generated tables

### `generated/tables/canonical_engine_benchmark.tex`

- Statement: all 36 cases pass independently at 64 and 96 nodes; errors/failures are
  exactly as tabulated.
- Source: `outputs/double_heston_benchmark/benchmark_summary.json`.
- Commit/ref: `BASE`.
- Classification: canonical implementation validation.
- Generator: reads JSON directly; no scientific value is edited.

### `generated/tables/real_market_development_comparison.tex`

- Statement: exact displayed calibration/holdout RMSE, IV RMSE, parameter count, and
  runtime for BS/Heston/DH.
- Source: model-comparison Markdown table in `docs/NTPC_SINGLE_STOCK_CALIBRATION.md`,
  whose values correspond to the hash-listed ignored `model_comparison.csv`.
- Commit/ref: `BASE`.
- Classification: development result; winner remains `NO_CLEAR_WINNER`.
- Boundary: not a real-market performance claim.

### `generated/tables/r2_representation_selection.tex`

- Statement: R2/R3 median best-start range-scaled parameter RMSE, median relative
  repricing RMSE, and frozen decision bands by noise level.
- Source: `evidence/g2_r2_r3_20260822/diagnostics_summary.json`;
  `final_decision.json`.
- Commit/ref: `BASE`.
- Classification: sealed representation-selection result selecting R2.
- Boundary: does not establish unique ten-parameter identification.

### `generated/tables/primary_comparison.tex`

- Statement: unified seed-mean neural metrics and traditional representative metrics;
  runtime row combines committed per-seed inference milliseconds with the committed
  traditional mean seconds.
- Source: `synthetic_test_comparison.csv`; `neural_seed_results.csv`;
  `runtime_metrics.json`; manifest hashes in
  `FINAL_EVALUATION_EVIDENCE_MANIFEST.json`.
- Commit/ref: `BASE`.
- Classification: canonical primary comparison on untouched test split.
- Boundary: no unique recovery claim; hardware-specific runtime is provenance.

### `generated/tables/completed_neural_noise.tex`

- Statement: Model1/Model2 seed-mean parameter RMSE, clean-latent normalized-price
  RMSE, and mean recovery rate at 0%, 0.10%, 0.25%, 0.50%, and 1.00%; population is
  1,250 test surfaces for each level and method.
- Source: `all_neural_seed_headline.csv` via explicit `git show`; summary manifest
  `neural/MANIFEST.json`.
- Commit/ref: read from `NOISE`; underlying evaluation commit
  `4ffddc2b9170ba494f75cedb2f5740cb806558c9`.
- Classification: completed neural observation-noise subset of the broader protocol.
- Boundary: excludes incomplete positive-level traditional calibration.

### `generated/tables/evidence_classification.tex`

- Statement: compact classification and source pointer for each major manuscript item.
- Source: manifests and documents listed above; generated from their machine-readable
  fields where available.
- Commit/ref: `BASE`, `NOISE`, or `MODEL3` as shown in each row.
- Classification: audit view; `RESULTS_INVENTORY.md` is more detailed.

## Generated figures

### `figures/speed_accuracy_tradeoff.pdf`

- Included file: `paper/figures/speed_accuracy_tradeoff.pdf`; deterministic copy of
  `paper/generated/figures/speed_accuracy_tradeoff.pdf`.

- Statement: relative position of mean evaluation time and known-truth parameter error
  for the three methods.
- Source: same three files as `primary_comparison.tex`.
- Commit/ref: `BASE`.
- Classification: visualization only; axes preserve source scale after log transform.
- Boundary: hardware/runtime differences are not scientific quality.

### `figures/fit_is_not_recovery.pdf`

- Included file: `paper/figures/fit_is_not_recovery.pdf`; deterministic copy of
  `paper/generated/figures/fit_is_not_recovery.pdf`.

- Statement: mean repricing nRMSE versus known-truth range-scaled parameter RMSE.
- Source: `synthetic_test_comparison.csv`.
- Commit/ref: `BASE`.
- Classification: visualization of the canonical identifiability-aware result.

### `figures/completed_neural_noise.pdf`

- Included file: `paper/figures/completed_neural_noise.pdf`; deterministic copy of
  `paper/generated/figures/completed_neural_noise.pdf`.

- Statement: seed-mean recovery-rate curves across completed neural noise levels.
- Source: `all_neural_seed_headline.csv` via pinned `git show`.
- Commit/ref: `NOISE`.
- Classification: completed neural-only robustness visualization.
- Boundary: no positive-noise traditional curve is plotted.

## Key manuscript claims

| Claim | Source | Commit/ref |
|---|---|---|
| Production pricer agrees with independent adaptive reference on 36 controlled cases | `benchmark_summary.json` | `BASE` |
| NTPC development comparison has no declared clear winner | NTPC document plus manifest status/winner rule | `BASE` |
| Practical non-identifiability is retained | G2 ambiguity/global diagnostics; R2/R3 final decision; primary conditioned metrics | `BASE` |
| R2 selected and frozen | `final_decision.json`; R2 contract | `BASE` |
| 10k dataset is complete, validated, and byte-identical on full replay | final 10k manifest and replay report | `BASE` |
| Traditional dominates synthetic-test accuracy but costs about 375.96 s/surface | primary comparison CSV/runtime JSON | `BASE` |
| Both networks produce valid vectors on 100% of test surfaces but have larger parameter errors than traditional | `synthetic_test_comparison.csv` | `BASE` |
| Low repricing does not certify parameter recovery | `identifiability_metrics.json`; conditioned rates quoted in Section 11 | `BASE` |
| Positive-noise neural evaluations are complete for both models/seeds/all levels | neural manifest/headline CSV | `NOISE` |
| Positive-noise traditional completion does not exist | cloud handoff status and missing final CSV statement | `NOISE` |
| No OOD, G8, or Model3 result exists | absence of completed-result artifact; explicit pending boundaries in R2/noise/Model3 sources | `BASE`, `NOISE`, `MODEL3` |

## Prohibited claims

Do not claim any of the following from this draft:

1. completed three-way positive-noise robustness;
2. an OOD/boundary result;
3. a G8 real-market result;
4. a Model3 training/pilot/research result;
5. unique recovery of ten parameters;
6. equivalence to unavailable teammate code;
7. physical-measure forecasting from risk-neutral option fit.

Any later evidence must update this inventory first, including its immutable source
hashes/commit and classification, before it may enter the paper.
