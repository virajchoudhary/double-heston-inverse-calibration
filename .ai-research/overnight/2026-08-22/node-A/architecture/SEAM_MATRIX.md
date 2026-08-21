# Canonical Seam Matrix — Node A, Overnight 2026-08-22

Status: PROVISIONAL ARCHITECTURE RECOMMENDATION (diagnostic run; no code changed).
All classifications trace to evidence in `../FINDINGS.md` (F1–F5) and direct source audit.

Legend: **KEEP** = canonical as-is · **ADAPT** = reuse the concept after re-homing onto canonical
contracts · **ISOLATE** = keep quarantined under archive import, no canonical path ·
**DEPRECATE** = recommend removal from canonical path · **RDD** = REQUIRES RESEARCH DECISION.

| # | Seam / contract | Canonical (Stack A) | Archive-2 (Stack B) | Classification | Why |
|---|---|---|---|---|---|
| 1 | Canonical parameter order | `src/constants.py:7-18`, `[k,θ,σ,ρ,v0]×2` slow-first | `[v0,κ,θ,σ,ρ]×2` fast-first (`transforms.py:9-20`) | A: **KEEP** · B: **ADAPT** (adapter only) | Double permutation (factor swap + reorder) verified; adapter proven exact by gradient permutation (F5). Positional interop forbidden. |
| 2 | Parameter bounds | structural validity only, no sampling box | sigmoid boxes incl. rho∈(-0.95,-0.05) | **RDD** (A box policy) · B: **ISOLATE** | A's map can emit OOD params (unbounded above); whether to constrain to reviewed sampling box is a fairness/OOD trade-off, not tonight's call. B's boxes are a *sampling* contract tied to negative-rho market belief — incompatible with canonical disk. |
| 3 | Factor identity | slow = lower kappa, listed first | factor1 = fast (kappa2≤kappa1 enforced) | A: **KEEP** · B: **ADAPT** via adapter | Same *relative* convention (slow<fast in kappa), opposite numbering. Semantics recoverable; identity must never be inferred from position across stacks. |
| 4 | Positivity | softplus+ε, hard by construction | sigmoid box, hard | A: **KEEP** · B: **ISOLATE** | Both valid constructions; A's is scale-free (no box dependency). |
| 5 | Feller condition | hard ceiling σ=0.995·√(2κθ)·sigmoid | absent (σ∈(0.05,1.5) unconstrained vs κθ) | A: **KEEP** · B: **ISOLATE** | B's space contains Feller-violating regions; any B-generated data reused canonically must be validity-filtered first. |
| 6 | Correlation disk | polar map, ρ_s²+ρ_f²<0.995², hard | none; per-factor boxes only; disk-violating combos reachable | A: **KEEP** · B: **ISOLATE** | B can produce invalid joint correlation matrices (ρ² sum up to 1.805). |
| 7 | Production pricer | frozen Gauss-Laguerre engine `src/double_heston.py` | n/a (uses own COS) | **KEEP** | Scientific source of truth; already independently benchmarked (repo history). |
| 8 | Differentiable torch pricer | `src/torch_double_heston.py` (GLQ mirror) | `src/dheston/pricing/heston.py` (COS) | A: **KEEP** · B: **ADAPT** (as independent cross-check only) | A = production at ~1e-15 (F5). B agrees 1e-12..1e-9 liquid region; useful independent verification asset, not a canonical dependency. |
| 9 | PDE physics | none yet (repricing consistency only) | PDE residual on pricer output | **RDD** (blocked on Node C) | B's residual is mathematically well-formed but constrains the *pricer*, not the network (F2 #3): near-vacuous gradient wrt weights. A genuine Model-3 PDE-informed variant needs a network-side PDE construction — an architecture decision, not a merge. |
| 10 | Surface representation | fixed 108-vector (9×6×2) + masks + ids + metadata | variable-length point cloud + masks | 108: **KEEP as PROVISIONAL** (G2 open) · pattern: **ADAPT** B's variable-length interface concept | Model layer already G2-decoupled (`input_size=features.shape[1]`, `expected_input_size()`). Recommend a surface-representation interface so a G2 grid change touches only data layer. |
| 11 | Dataset contract | `SurfaceParameterDataset` (features/targets/masks/ids/metadata) | `SurfaceDataset` (padded point batches) | A: **KEEP** · B: **ISOLATE** | A preserves surface identity + finite checks; B couples to real-row schema. |
| 12 | Synthetic split | disjoint index splits, enforced (`train_pinn.py:46`) | template-resampled synthetics | A: **KEEP** | Train-only standardizer fit (`train_pinn.py:58-59`), no leakage found. |
| 13 | Real-market split | post-freeze evaluation only (policy) | chronological split + `verify_zero_leakage` | B concept: **ADAPT** for the frozen evaluation stage | Chronological + leakage verification is the right pattern for the eventual real-market holdout; must run on frozen models. |
| 14 | Real-market training rule | prohibited (research control) | `real_finetune` mode + `--continuous` weight updates on real data | B paths: **DEPRECATE / REMOVE FROM CANONICAL PATH** | Direct violation of canonical control (F3). If a real-finetuning ablation is ever authorized, it must be a separate explicitly-labeled experiment, never a named mode in a canonical entry point. |
| 15 | ANN baseline | `models/ann_model.py` + `src/train.py` | n/a | **KEEP** | Control arm for the eventual comparison. |
| 16 | Inverse physics network | `PhysicsInformedInverseCalibrator` + `DoubleHestonConstraintMap` | `DeepSurfaceInverseModel` | A: **KEEP** (Model 2 core) · B: **ISOLATE** | A's constraint map is the canonical structural-validity seam; B's network couples pooling + box constraints to archive semantics. |
| 17 | Evaluation metrics | `evaluate_parameters` / `evaluate_repricing` | in-trainer metrics | A: **KEEP** · B: **ISOLATE** | Canonical metrics separate recovery from repricing (required by research control). |
| 18 | Checkpoint policy | best-validation checkpoint (`train_pinn.py:81`) | per-epoch continuous real-mode checkpoints | A: **KEEP** · B continuous: **DEPRECATE** with #14 | Validation-gated checkpointing aligns with model-selection policy. |
| 19 | Reproducibility | `set_deterministic_seed`, seeded loaders | seeded synthetics; unbounded continuous mode | A: **KEEP** · B continuous: **DEPRECATE** with #14 | B's continuous mode has no bounded run definition (non-reproducible by design). |
| 20 | Status-document ownership | `docs/RESEARCH_CONTROL_AND_CURRENT_STATUS.md` | none | **KEEP** | Canonical gate ledger; only human-approved changes (Phase J plan). |

## Derived canonical seam (the one-sentence architecture)

> The canonical inverse-calibration architecture is: canonical parameter contract
> (`src/constants.py`) + frozen production pricer + its validated torch mirror
> + `DoubleHestonConstraintMap` structural validity + fixed-size surface features behind a
> representation interface + synthetic-only primary training with validation-gated
> checkpointing + post-freeze real-market evaluation (chronological, zero-leakage-verified).
> Archive-2 contributes: the variable-length surface pattern, the chronological/zero-leakage
> evaluation pattern, and the COS pricer as an independent cross-check — all behind adapters,
> never positional. The PDE-informed tier (Model 3) is **not** achievable by importing Stack
> B's residual; it requires a research decision on a network-side PDE construction.

## G2-resilience note (Phase H)

No magic `108` exists in model code: `input_size` flows from `dataset.features.shape[1]`
(`src/train.py:204`, `src/train_pinn.py:326`, `src/run_pinn_synthetic_baseline.py:61`) and
`expected_input_size()` derives it from grid constants (`src/surface_grid.py:86`). G2 grid
changes therefore touch only: grid constants, synthetic generation, and dataset
checkpoints — not model classes. Remaining coupling to audit at G2 time: metadata/config
records that log `input_size` (fine) and any serialization assuming fixed dims.
