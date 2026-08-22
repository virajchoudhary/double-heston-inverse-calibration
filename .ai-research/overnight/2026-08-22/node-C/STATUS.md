# Node C Status — Overnight 2026-08-22

Role: PDE_PHYSICS_AUDIT. Branch `overnight/20260822-c-pde` from genesis
`642702e6706a3d17b3031619f35bda39bc144483`.

## Checkpoint log

- ~00:50 IST: bootstrap, phases A/B complete (derivation + independent
  cross-derivation), probe evidence, 25-case suite, FINDINGS/STATUS/tables,
  first push (a44c8bb), final report v1 (861f232), issue #18 report
  (be1eab7).
- ~01:25 IST: extension probe — F2 closed BIT-EXACTLY (production loss ==
  broken operator), F3 quantified (correct wiring 4e-9 vs broken 7e-2),
  defect invariance, market-price independence, BS-limit O(sigma^2),
  terminal-condition sweep, in-grid negative-price scan; peer verification of
  Node A F16/F18/G2 cross-check (751551a, faaeaf3).
- ~01:40 IST: adversarial review returned — all load-bearing claims
  CONFIRMED (independent re-execution); 5 corrections applied; F13 added
  (dead FourierConfig fields, canonical-stack autograd scan CLEAN, probe
  float32 fix, latent tau=0 early-return, dead compute, naming); broadened
  120-point certification sweep (<= 1.6e-8); independent derivation +
  adversarial report committed as artifacts; correct-wiring regression test
  added — suite now 27/27 (54fb52e, 27a6a25). Issue #18 correction comment
  re dead fields posted.
- ~01:55 IST: cross-interpreter datapoint — repo's own canonical-engine tests
  34/36 on py3.9 + documented shim (2 explained environment artifacts:
  fixture atol=1e-12 missed by a 3.3e-12 cross-platform libm diff; the test
  file's own zip(strict=True)); variance-mean propagation gap quantified
  (F15: -1.1% at 7d to -14.5% at 90d — per-date v0 fitting vindicated);
  ledger timestamps corrected to wall clock. Node A integrated the Node C
  refresh (their F19/F23) and corrected the dead-field attributions;
  convergence recorded (their F20 / my F14).
- ~03:15 IST: Node B landed — verified (F16): global ambiguity replicates on
  the full 108 grid (12/12 starts); factor-swap degeneracy exact to 4.26e-14
  on the production pricer (bitwise on their fast pricer; float association
  order explains the difference); PDE residual cannot resolve the
  near-equivalent manifold (residual floor 4e-9 vs near-equivalence price
  RMSE 1.1e-6). Answer posted to issue #18; Node B's final report adopted the
  "physics regularises, does not identify" formulation.
- 03:52-05:52 IST: polls 3-6 — both peers stable at final states since
  03:55 (Node B final f0844c5; Node A 5d54735). Node B independently hit the
  same platform-sensitive fixture tolerance failure (3.3e-12 vs 1e-12) on
  py3.13 — matches Node C's py3.9 artifact (F-report: cross-platform
  tolerance, not a regression).

## FINAL (05:55 IST consolidation)

- Focused suite: 27/27 PASS. Evidence directory 216K, no binaries, secrets
  scan clean (single false positive = this file's own checklist line).
- Branch delta vs genesis: 19 files, +3799 lines — tests + .ai-research
  evidence only; src/, models/, configs/, docs/ untouched.
- Final safety: origin/main == 642702e (verified 05:12 and at close); no
  force push; no 10k generation; no neural training of any kind; no
  real-market fine-tuning executed; G2 untouched; no environment mutation;
  production pricer unmodified. All Node C evidence pushed to
  overnight/20260822-c-pde only.

## Phase status

| Phase | Status | Key output |
|---|---|---|
| Git bootstrap | DONE | clean tree, genesis verified, branch created from genesis, evidence dirs created |
| A. Stochastic specification freeze | DONE | `derivations/CANONICAL_DOUBLE_HESTON_PDE.md` §1 |
| B. Canonical Feynman-Kac PDE | DONE | derivation + independent cross-derivation + numerical certification (rel. residual <= 1.3e-15) |
| C. Terminal/boundary conditions | DONE | derivation §3; Archive-2 "boundary" term classified as not boundary physics |
| D. Archive-2 PDE loss line-by-line | DONE | `tables/ARCHIVE2_PDE_TERM_MAP.md` — INCORRECT (all variance derivatives structurally zero) |
| E. Canonical model classification | DONE | constraint + differentiable-repricing informed inverse network (NOT PDE-informed) |
| F. Parameter contract matrix | DONE | `tables/PARAMETER_CONTRACT_MATRIX.md` — ADAPTER + SEMANTIC CONFLICT |
| G. Constraint audit | DONE | Archive-2 emits canonical-invalid vectors (reproducible, 3 violations) |
| H. Limiting-case tests | DONE | additivity 1.8e-15; half-factor reduction 3.2e-11 vs independent COS; BS limit 4e-4; parity 7e-15 |
| I. PDE residual sanity tests | DONE | machine-precision residual on canonical pricer + coefficient-perturbation sensitivity + Archive-2 zero-derivative proof |
| J. Training/validation objective audit | DONE | Archive-2 validation excludes PDE loss (pde_points=0 + no_grad; artifact valid_pde=0.0); canonical stack has no mismatch |
| K. Real-market training policy | DONE | Archive-2 real_finetune updates all NN weights -> conflicts with control; recommend ISOLATE + DISABLE BY DEFAULT |
| Web/literature check | DONE (light) | standard references cross-check only; no equation copied |
| Peer sync | DONE (one fetch) | Node A FINDINGS corroborates (their candidates = my proven items); Node B branch at genesis, no evidence yet |
| Final report | DONE | `FINAL_REPORT.md` |

## Test evidence

- `tests/test_node_c_pde_physics_audit.py` — 27 cases, 27 PASS
  (runner: `tests_evidence/run_node_c_tests.py`, results
  `tests_evidence/pytest_equivalent_results.json`; interpreter lacks pytest,
  no environment mutation per compute policy).
- Numerical probes: `tests_evidence/probe_residuals.py` ->
  `probe_results.json`; `tests_evidence/probe_extensions.py` ->
  `probe_extension_results.json`.
- Adversarial review artifact: `tests_evidence/ADVERSARIAL_REVIEW_REPORT.md`
  (independent re-derivation + re-execution; all load-bearing claims
  confirmed).
- Independent cross-derivation committed:
  `derivations/INDEPENDENT_CROSS_DERIVATION.md`.
- Reproducibility: full suite + probes re-run from a clean worktree of the
  pushed branch — bit-identical values.

## Safety confirmations

main untouched; no force push; no 10k generation; no real-market training; G2
untouched; no environment mutation; production pricer unmodified (evidence
tests import it read-only; the only loader shim is a documented Python-3.9
typing compatibility exec inside the test harness, zero mathematical lines
changed); branch pushed only to `overnight/20260822-c-pde`.
