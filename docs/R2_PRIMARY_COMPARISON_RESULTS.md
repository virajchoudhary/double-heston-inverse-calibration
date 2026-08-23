# R2 Primary Comparison — Results Reconciliation

Status: IN_PROGRESS (evidence generation running).
Protocol: `docs/R2_PRIMARY_COMPARISON_PROTOCOL.md` (frozen before training at commit
`5ea9fd0`, remote-verified; Issue #32).
Dataset: `data/final_r2_clean_10000/` — surfaces SHA-256 `148b579a…f1f6`,
splits 7,500 / 1,250 / 1,250 stored per surface.
Evidence bundle: `evidence/r2_primary_comparison_20260823/`.

This document reconciles the frozen metric families into the primary
scientific narrative. It is filled in only from the committed evidence
bundle; no number in this document may come from anywhere else.

## 1. Execution chronology

| Step | Artifact |
|---|---|
| Protocol freeze (pre-training) | commit `5ea9fd0` (pushed, remote-verified); Issue #32 |
| Implementation repair | commits `ab11b09`, `9bf1d49` |
| Smoke runs (`DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT`) | `checkpoints/r2_primary_comparison/smoke/`, `smoke_calibration_starts.csv` |
| Research training Model 1 seeds 11/22/33 | `checkpoints/r2_primary_comparison/model1_seed*/` |
| Research training Model 2 seeds 11/22/33 | `checkpoints/r2_primary_comparison/model2_seed*/` |
| Traditional calibration (frozen, test split) | `traditional_calibration_starts.csv` + summary |
| Final frozen evaluation | metric JSONs + `synthetic_test_comparison.csv` |

## 2. Training manifest (filled from training_run_manifest.json)

TBD

## 3. Unified comparison (filled from synthetic_test_comparison.csv)

TBD

## 4. Findings by frozen metric family (filled from metric JSONs)

### 4.1 Parameter recovery TBD
### 4.2 Constraint validity TBD
### 4.3 Repricing TBD
### 4.4 Identifiability-aware interpretation TBD
### 4.5 Stability TBD
### 4.6 Runtime TBD

## 5. Claim-discipline reconciliation

TBD — every allowed claim checked against evidence; forbidden claims
explicitly restated as not made.

## 6. Limitations

TBD
