# OOD Boundary Evaluation Readiness Seal

Status: **EVALUATION_INFRASTRUCTURE_SEALED_NO_RESULTS**, 25 August 2026.
Config: [../configs/ood_boundary_evaluation_ready.yaml](../configs/ood_boundary_evaluation_ready.yaml).
CLI: `scripts/run_ood_boundary_evaluation.py`.

This milestone adds execution infrastructure only. It does not compute,
display, or store any Model1/Model2/Model3/traditional result on the frozen
420-surface OOD benchmark. The prior cohort seal remains byte-identical.

## Verified frozen input

- Branch: `research/ood-boundary-protocol`; starting/final protocol evidence at
  `b6c5e5d0c60d5a99d767ebb3db5175859f310293` before this infrastructure commit.
- Clean research surfaces: 360 (120 boundary, 120 distribution shift, 120
  maturity/conditioning shift), all with exactly 20 usable slots.
- Incomplete surfaces: 60, twelve each of five deterministic mask patterns and
  at least ten usable slots.
- Total immutable research order: 420 unique surface IDs.
- Development sanity artifact: exactly 12 parameter rows, zero surfaces.
- Cohort pricing failures: zero.
- Replay: every scientific artifact hash matched; replay is
  `VERIFIED_IDENTICAL`.

No prediction or method-metric file exists under either immutable cohort tree.
The 12-row sanity panel was not used for inference because it has no surfaces
and its truths were drawn from the first selected research candidates; deriving
surfaces from it would leak a subset of research outcomes into development.

## Execution lock

The default command can use only an explicitly generated nonresearch fixture.
Frozen research evaluation requires all of:

```text
run --cohort research
--authorize-frozen-evaluation
--confirmation AUTHORIZE_FROZEN_OOD_RESEARCH_EVALUATION_V1
```

An environment variable alone cannot authorize loading. The loader first checks
the three explicit choices, then verifies the sealed identity manifest, cohort
SHA-256, and exact surface-ID ordering. The authorization mechanism—not the raw
phrase—is recorded in the future result manifest. Output creation is refused
under the frozen cohort/replay trees or completed primary dataset directories.

## Reused sources of truth

| Operation | Canonical implementation |
|---|---|
| R2 validation/features | `src/r2_primary/dataset.build_r2_features`, R2 serialization |
| Model checkpoint load | `src/r2_primary/training.load_run` |
| Neural inference | `src/r2_primary/training.predict_parameters` |
| Parameter metrics | `parameter_recovery_metrics` |
| Validity metrics | `constraint_validity_metrics` |
| Production repricing | `price_double_heston_surface` via `reprice_normalized` |
| Repricing metrics | `repricing_metrics` |
| Identifiability diagnostics | `identifiability_aware_metrics` |
| Traditional starts | `calibrate_double_heston` with frozen settings |
| Representative rule | lowest loss, then lowest start index |

No metric formula was reimplemented in a competing framework. Reference ranges
and standard deviations come only from the frozen R2 train truth split; they are
truth-panel metadata, not model results.

## Method readiness

- **Model1:** identity-gated adapter implemented for seeds 11/22/33. Current
  local status is blocked because `.pt` checkpoints are untracked and absent.
- **Model2:** same adapter/checkpoint policy as Model1. No training, retuning,
  seed substitution, or partial-seed evaluation is permitted.
- **Model3:** contract-only. Status remains
  `WAITING_FOR_FROZEN_RESEARCH_CHECKPOINTS`. No Stage-A execution, Stage-B
  training, pilot/smoke weights, fake predictions, or current foundation output
  may enter OOD evaluation. A future manifest must contain approved checkpoint
  path/hash, Git/config/data identities, seeds, canonical parameter contract,
  inference interface, and explicit OOD approval. Requesting Model3 while it is
  unavailable makes the run `PARTIAL_OR_BLOCKED`; it can never be omitted from
  the completion calculation when requested.
- **Traditional:** deterministic 60-row subset is materialized without
  evaluation: evenly spaced indices starting at zero within each of four active
  cohorts, 15 rows per cohort. Execution supports crash-safe JSONL journaling,
  resume, retained failures, exact frozen settings, and no altered retry.

For incomplete observations, neural features retain canonical shape and exact
zero/mask semantics. Traditional calibration receives only legally observable
quotes; masked zeros are filtered before calibration, never imputed. The primary
incomplete repricing diagnostic reprices predictions against their retained
clean parents; the parent is never supplied as model input. An observed-slot
diagnostic excludes masked slots.

Future frozen-research metric artifacts include per-cohort summaries,
hash-pinned ID-baseline degradation ratios, 2,000-resample bootstrap intervals
(seed 20260829), and traditional start/surface success and failure rates.
If a bootstrap interval spans the materiality boundary, or if more than five
percent of traditional starts fail, the top-level result is explicitly
`INCONCLUSIVE_*`, never accepted as complete.

## Commands available now

```bash
python -m scripts.run_ood_boundary_evaluation verify-freeze
python -m scripts.run_ood_boundary_evaluation prepare-readiness
python -m scripts.run_ood_boundary_evaluation run-development-smoke --methods truth_pipeline,model1,model2,model3 --output evidence/ood_boundary_development_smoke_v1
```

Development smoke uses three newly generated fixture surfaces unrelated to the
frozen truths. Missing Model1/Model2 checkpoints produce explicit blockers;
Model3 produces `WAITING_FOR_FROZEN_RESEARCH_CHECKPOINTS`; nothing is faked.
The optional one-start traditional smoke uses `max_nfev=1`, is marked
non-comparable, and never touches research rows.

## Future locked commands

Do not execute these during this milestone:

```bash
# Neural-only frozen OOD evaluation
python -m scripts.run_ood_boundary_evaluation run \
  --cohort research \
  --methods model1,model2 \
  --authorize-frozen-evaluation \
  --confirmation AUTHORIZE_FROZEN_OOD_RESEARCH_EVALUATION_V1 \
  --output evidence/ood_boundary_research_results_neural_v1

# Add the preselected traditional subset after separate scheduling approval
python -m scripts.run_ood_boundary_evaluation run \
  --cohort research \
  --methods model1,model2 \
  --include-traditional \
  --workers 10 \
  --authorize-frozen-evaluation \
  --confirmation AUTHORIZE_FROZEN_OOD_RESEARCH_EVALUATION_V1 \
  --output evidence/ood_boundary_research_results_complete_v1
```

A later Model3 invocation may add `,model3` only when its approved frozen
manifest exists and passes runtime validation.

## Compute plan

- Model1/Model2 inference: local CPU is sufficient; batched amortized inference
  is expected to be seconds-to-minutes for 420 rows per seed.
- Model3: waiting; no compute recommendation until its final architecture and
  checkpoint contract exist.
- Traditional: repository measurement for the frozen budget was 124–170 seconds
  per surface. Sixty surfaces imply roughly 2.1–2.8 single-core hours; ten
  surface-level workers imply roughly 13–17 wall minutes plus process startup.
  Long multiprocessing is intentionally not started tonight.

## Result intake

Every run writes strict artifact hashes, requested methods/statuses, protocol/
evaluation config hashes, cohort/file/order hashes, Git SHA, environment,
hardware, authorization record where applicable, and completion status.
`COMPLETE` is refused for partial or blocked runs. A research manifest without
the exact recorded authorization mechanism cannot pass intake. Runtime artifacts
are excluded from deterministic core replay because wall time is provenance,
not repeatable scientific content.
