# G2 R2-vs-R3 Representation Selection — Final Results

Status: EXECUTED AND SEALED — 22 August 2026

This document records the one-time execution of the predeclared self-governed
G2 representation-selection experiment defined in
[G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md).
The protocol, seeds, thresholds, interpretation bands, and stopping rule were
frozen before any R2/R3 outcome was computed and were not modified afterwards.

## 1. Question

Which market-supported representation — R2 (two ranked expiries) or R3 (three
ranked expiries, masked) — is the most defensible primary representation, and
how much practical ambiguity remains after it is frozen? This is a
representation-selection question, not a claim of universal unique ten-parameter
identification.

## 2. Frozen protocol reference

- Protocol: `docs/G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md` (merged at
  `c64aaaf` on `main`, before execution).
- Frozen randomization: truth-selection seed `20260822`, multi-start seed
  `20260823`, noise base seed `20260824`.
- Interpretation bands (strong: ≥25% improvement in both median and maximum
  dispersion AND fewer separated clusters; partial: ≥10% in both with no cluster
  increase), stopping rule, and completion labels: protocol sections 7–8,
  implemented in `src/g2_r2r3/frozen.py` and `src/g2_r2r3/decision.py`.
- Holdout guardrail: the existing 5% real-market holdout-deterioration ceiling
  applied to a design holding out the ±0.10 wings; both R2 and R3 calibrate on
  those wings, so no directly comparable holdout metric exists. Recorded as
  **NOT APPLICABLE** per the protocol rather than replaced with a substitute.

## 3. Git and environment provenance

| item | value |
|---|---|
| expected/verified starting `main` | `c64aaaf54f72a48768316c582d312dd8cf27a089` |
| execution branch | `research/g2-r2-r3-selection` |
| truth-panel freeze (checkpoint A) | `6986281` |
| HARNESS_READY (checkpoint B) | `1aad179` |
| market audit (checkpoint C) | `0920f87` |
| matrix + decision (checkpoint G) | `09aa1d7` |
| Python | 3.13.4 (Windows 11, CPU-only) |
| NumPy / SciPy / pandas / torch | 2.2.6 / 1.16.2 / 2.3.2 / 2.11.0+cpu |
| git | 2.51.0.windows.1 |

The SciPy 1.16.2 pin is load-bearing: the reviewed sampler's
`qmc.LatinHypercube(seed=...)` stream is version-sensitive, and the committed
`evidence/g2_r2_r3_20260822/truth_panel.csv` is the byte-exact frozen artifact.
The frozen production pricer `src/double_heston.py` and every canonical
contract (parameter order, constraints, bounds) are untouched by this study;
the fast diagnostic pricer (`src/g2_r2r3/pricer.py`, adapted with provenance
from the archived Node-B toolkit, not merged) was validated against the
production pricer at worst max-abs difference **0.0** across all 20 truths ×
R2/R3 before use.

## 4. R2 definition

First TWO eligible listed expiry ranks × central log-moneyness
`[-0.10, -0.05, 0.00, +0.05, +0.10]` × calls and puts = 20 nominal price slots,
spot-normalized prices, actual time-to-maturity supplied explicitly,
existing per-rank rate/carry conditioning, no interpolation or extrapolation.

## 5. R3 definition

Same contract with the first THREE eligible listed expiry ranks = 30 nominal
slots, explicit masks for unsupported/unusable observations, missing
observations never filled with model prices. Per the protocol, R3 is a
maximum-size masked representation, not a claim that real surfaces contain 30
equally reliable observations.

## 6. NTPC market-support table

All five development dates under the existing official-NSE
support/activity/quote-selection contract (raw UDiFF archives; Hungarian
assignment with the 0.05 target gate; activity, moneyness, and Black-IV
eligibility on the matched-futures forward). Rates use the committed
hash-sealed RBI 91-day T-bill observations (5.2521% on 07-01, 5.3324% on
07-15), carried forward to 07-08, 07-22, and 07-29 by that contract's own convention (the
07-22 carry-forward was already the committed convention; 07-08/07-29 lack preserved
auction artifacts) — no new acquisition, nothing fabricated.

| date | eligible ranks (expiry, DTE) | R2 usable | R3 usable | R2 compl. | R3 compl. | dominant failure |
|---|---|---:|---:|---:|---:|---|
| 2026-07-01 | 1–3 (07-28/27, 08-25/55, 09-29/90) | 11/20 | 11/30 | 0.55 | 0.37 | rank-3 activity; 9 quote-gate misses |
| 2026-07-08 | 1–3 (07-28/20, 08-25/48, 09-29/83) | 18/20 | 18/30 | 0.90 | 0.60 | rank-3 activity; 2 quote-gate |
| 2026-07-15 | 1–3 (07-28/13, 08-25/41, 09-29/76) | 19/20 | 19/30 | 0.95 | 0.63 | rank-3 activity; 1 quote-gate |
| 2026-07-22 | 1–3 (07-28/6, 08-25/34, 09-29/69) | 18/20 | 18/30 | 0.90 | 0.60 | rank-3 activity; 2 quote-gate |
| 2026-07-29 | 1–3 (08-25/27, 09-29/62, 10-27/90) | 12/20 | 12/30 | 0.60 | 0.40 | rank-3 activity; 8 quote-gate |
| **aggregate** | 13 distinct DTEs (6–90) | **78/100** | **78/150** | **0.78** | **0.52** | — |

Key finding: the third expiry rank contributes **zero** usable central-five
slots on every date — far-month NTPC option chains are wholly inactive under
the existing activity contract (verified directly against the raw UDiFF files;
e.g. 2026-09-29 on 07-15: 26 listed rows, 0 active). The per-date R2 counts
reproduce the committed three-date inventory exactly (11/19/18 of 20). All
five dates remain DEVELOPMENT / DIAGNOSTIC and permanently excluded from final
G8.

## 7. Synthetic truth panel

20 deterministic truths frozen at checkpoint A before any outcome: the four
standing representative G2 cases (`case_1`–`case_4`, exact committed maximin
selection) plus 16 reviewed-interior truths — the first 16 accepted rows of a
single 64-row `interior_train` draw with seed 20260822 through the existing
reviewed sampler and its own margin gate. No new parameter distribution. Each
truth is assigned an actual date conditioning profile by `truth_index mod 5`
(actual DTEs, continuous rates, futures-implied carries), identical for R2 and
R3.

## 8. Jacobian / local-information comparison

Range-scaled Jacobians of spot-normalized prices (central differences,
validity-aware 1e-4 relative step), SVD diagnostics at the existing 1e-6
practical-rank tolerance, computed at each truth on its own profile:

| metric (median over 20 truths) | R2 | R3 |
|---|---:|---:|
| condition number | 1.84e8 | 1.54e6 |
| smallest singular value | 1.13e-9 | 2.24e-7 |
| practical rank distribution | 7×10, 8×8, 9×2 | 8×1, 9×12, 10×7 |

R3 is roughly two orders better conditioned and more often locally full rank.
Under the protocol this is supporting evidence, not a winner switch.

## 9. Clean (0% noise) calibration comparison

| dispersion over near-equivalent 12-start sets | R2 | R3 |
|---|---:|---:|
| median pairwise separation (median of cells) | 0.6924 | 0.4498 |
| maximum pairwise separation (max over cells) | 1.5160 | 1.0495 |
| mean separated clusters | 6.20 | 4.45 |
| cells with multiple basins | 20/20 | 19/20 |
| median boundary-hit rate (near-equivalent set) | 0.17 | 0.00 |
| median best parameter RMSE (range-scaled) | 0.154 | 0.046 |

On clean data R3 meets the **strong** improvement band (median −35.0%, maximum
−30.8%, clusters 6.20 → 4.45). The two representations are not
uniquely invertible even clean — multi-basin ambiguity persists — but the third
expiry genuinely adds clean information.

## 10. Noise comparison (0.5%, 1%, 2%)

Predeclared primary comparison level: **0.5%** (smallest realistic noise; the
predeclaration anchor is `frozen.NON_IDENTIFIABILITY_NOISE_LEVEL = 0.005`, committed at
checkpoint A before any outcome).
Median best-start range-scaled parameter RMSE / median best relative repricing
RMSE:

| noise | R2 param RMSE | R3 param RMSE | R2 repricing | R3 repricing |
|---|---:|---:|---:|---:|
| 0.5% | 0.383 | 0.356 | 0.884% | 0.735% |
| 1.0% | 0.438 | 0.340 | 1.982% | 1.456% |
| 2.0% | 0.478 | 0.389 | 2.592% | 2.946% |

Comparative assessment by level (frozen bands on median/max dispersion with
cluster condition):

| level | classification | median impr. | max impr. | clusters R2→R3 |
|---|---|---:|---:|---|
| 0% | STRONG_IMPROVEMENT | +35.0% | +30.8% | 6.20 → 4.45 |
| 0.5% | NO_MATERIAL_IMPROVEMENT | **−17.8%** | +1.7% | 6.60 → 5.50 |
| 1% | NO_MATERIAL_IMPROVEMENT | +3.4% | −6.3% | 5.75 → 5.35 |
| 2% | NO_MATERIAL_IMPROVEMENT | +16.1% | −1.5% | 6.25 → 4.65 |

The clean advantage of R3 does not survive realistic noise: at 0.5% the median
near-equivalent dispersion is actually worse for R3 (0.93 → 1.10) and at every
noisy level at least one dispersion measure fails the ≥10% band. The decision
is therefore robust across all realistic noise levels, not just the primary
one.

## 11. Boundary / validity behavior

- All recovered vectors are structurally valid by construction (latent
  constraint transform); `boundary_diagnostics` proximity flags are recorded
  per run.
- Boundary-hit rate among near-equivalent solutions: ~0.17 (R2) / 0.00 (R3)
  clean, rising to ~1.00 for both at every noisy level — at market noise the
  near-equivalent set presses against the hard envelope for both
  representations.
- Optimizer behavior (all 1,920 runs retained): SciPy status 0 (max-nfev
  exceeded) 1,186 runs; gtol/xtol/ftol terminations 721; recorded exceptions
  10 (2 R2, 8 R3 — all retained with messages). Reached-cap rate 0.59 (R2) /
  0.65 (R3) under the frozen 120-evaluation budget.

## 12. Near-equivalent cluster behavior

Complete-linkage clustering (cutoff 0.10 on range-scaled coordinates) over the
near-equivalent set (`max(1.05 × best, 2.5e-7)` repricing RMSE threshold):

- Both representations show multiple materially separated near-equivalent
  basins in essentially every cell at every noise level (19–20 of 20 cells).
- Median pairwise separations of 0.45–1.15 range-scaled units mean alternative
  parameter vectors spanning a large fraction of the reviewed parameter box
  reprice the observation set within tolerance.
- R3 has somewhat fewer clusters (5.5 vs 6.6 mean at 0.5%) but the frozen bands
  require the dispersion improvements to accompany that, which they do not at
  any noisy level.

## 13. Factor-swap result

Swapping slow/fast blocks is a degeneracy of the pricing map to near-machine
precision: worst max-abs price difference **2.8e-14** across all 20 truths
(exact zero under constant carry; market per-rank carry introduces only
floating-point summation-order noise). The canonical `kappa_slow < kappa_fast`
ordering convention rejects the swapped twin in **20/20** cases
(`enforce_ordering=True`), confirming the declared tie-breaking behavior.

## 14. Runtime

Median per start: 10.2 s (R2) / 15.9 s (R3) (median of per-cell medians) on CPU (12-core Windows, 6 shard
workers). Total optimizer time across the matrix: R2 9,771 s, R3 15,378 s
(~7.0 hours combined). Full evidence, including per-run runtimes, is preserved
in `synthetic_runs.csv`.

## 15. Limitations

1. **R3's third expiry is unusable on the development market panel.** Its extra
   slots are 100% masked on all five dates (aggregate R3 completeness 0.52 vs
   R2 0.78). The predeclared hard requirements define market support as
   reproducible construction with explicit masking, which R3 satisfies; the
   protocol explicitly defines R3 as a maximum-size masked representation. The
   frozen rule was therefore applied as written — but a representation whose
   added slots are entirely masked in current NTPC market practice would have
   delivered no additional real observations had it been selected.
2. **Practical non-identifiability persists under every candidate and noise
   level**: surfaces are fitted to noise scale while best-start parameters
   displace by ~0.36–0.48 range-scaled RMSE (≫ the 0.05 material-displacement
   convention).
3. Rates for 07-08 and 07-29 are carry-forward observations (documented
   artifact absence), affecting only the IV eligibility gate and synthetic
   conditioning, not selection outcomes.
4. The clean-data STRONG improvement for R3 is real but is not "practical
   information" under the predeclared primary comparison; it is retained here
   as evidence that the marginal third expiry adds clean information that
   market-scale noise destroys.
5. Optimizer budget (120 evaluations, the committed G2-ambiguity convention)
   leaves a ~60–65% reached-cap rate; no budget escalation was attempted
   because optimizer-only work is closed and the predeclared arms are
   identical across candidates.
6. The holdout guardrail is NOT APPLICABLE to this comparison by design (no
   directly comparable holdout metric exists for R2/R3).

## 16. Exact application of the predeclared decision rule

Applied once, mechanically, by `src/g2_r2r3/decision.py` with thresholds
imported only from `frozen.py`:

1. Hard market-construction requirements: **both R2 and R3 satisfy** them
   (explicit masking, no interpolation/extrapolation, reproducible official-NSE
   construction, canonical contracts unchanged, G8 protection intact).
2. Rule 1 (freeze R3 if strong/partial practical-information improvement
   without holdout violation): **fails** — classification at the predeclared
   0.5% level is NO_MATERIAL_IMPROVEMENT (and at 1% and 2%).
3. Rule 2: **freeze R2** as the simpler market-supported representation.
4. Rule 3: both candidates remain practically non-identifiable at realistic
   noise (median best parameter RMSE 0.383 ≫ 0.05 while repricing stays within
   2× the noise level), so `PRACTICAL_NON_IDENTIFIABILITY =
   RETAINED_RESEARCH_FINDING`. No unlimited representation search is reopened.
5. Rule 4 (fail both) does not apply.

## 17. Selected representation

**R2 — ranked two-expiry central-five calls/puts (20 slots), spot-normalized,
actual maturity conditioning, existing rate/carry conditioning.**

## 18. Exact G2 completion label

```text
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
```

## 19. Retained research finding

```text
PRACTICAL_NON_IDENTIFIABILITY = RETAINED_RESEARCH_FINDING
```

At 0.5% noise, median best-start parameter displacement is 0.383 range-scaled
RMSE (7.7× the material-displacement convention) while the fitted surface
matches observations to 0.88% relative RMSE — repricing quality is not
parameter-recovery evidence, and parameter-recovery claims must remain
tolerance/equivalence-class conditioned. R3's better conditioning and clean
recovery show the information deficit is partly representational, but the
residual ambiguity at market tolerance is intrinsic to the observation scale,
not removable by the third expiry.

## 20. Exact next project step

`FORMALIZE THE FROZEN R2 REPRESENTATION INTERFACE, THEN REGENERATE/REVALIDATE
THE FINAL SYNTHETIC SURFACE CONTRACT` — per the control document's post-G2
sequence: formalize the R2 interface; regenerate and validate the final
synthetic dataset; freeze splits; then run the traditional-calibration vs
Model-1 ANN vs Model-2 informed-inverse comparison under identical evaluation
contracts; final 10k generation, ANN/Model-2 training, final G8 date selection,
and real-market evaluation remain separately controlled milestones that were
NOT started here.

## Reproducing this study

```bash
git checkout research/g2-r2-r3-selection   # checkpoint G commit 09aa1d7 or later
python scripts/run_g2_r2r3_truth_panel.py      # checkpoint A (panel freeze)
python -m pytest tests/test_g2_r2r3_harness.py # 18/18 at HARNESS_READY; 19/19 incl. audit-added integration test
python scripts/run_g2_r2r3_smoke.py            # tiny smoke
python scripts/run_g2_r2r3_market_support.py   # checkpoint C (5-date audit)
python scripts/run_g2_r2r3_matrix.py           # 1,920-run matrix (shardable)
python scripts/run_g2_r2r3_merge.py            # validate + merge run log
python scripts/run_g2_r2r3_diagnostics.py      # diagnostics + frozen decision
```

Raw evidence: `evidence/g2_r2_r3_20260822/` (manifest, truth panel, market
audit, per-run JSONL/CSV, Jacobian/dispersion records, noise summaries,
diagnostics summary, final machine-readable decision). The protocol document
is the pre-result contract and was not rewritten.

## Adversarial review

Three independent audits ran before the decision was applied
(representation-contract/leakage, decision-rule fidelity, determinism/seeds —
full trail in the manifest): the material findings (a run-record column-name
mismatch that would have crashed diagnostics; caller pre-guess of the selected
representation) were fixed with integration tests before the decision ran; the
rule-1 market-masking tension is preserved as limitation 1 rather than resolved
by editing the frozen rule. No threshold, seed, band, or rule order changed
after outcomes.
