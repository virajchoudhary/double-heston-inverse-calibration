# Residual-connection forgetting test — results

Design: `DESIGN.md` (written before any result). Numbers: `final_results.json`, produced by
`scripts/final_report.py`. 3 arms × 3 seeds, every checkpoint scored on 24 fresh truths
(seed 913311). 0 fit failures.

## Answer

**Residual connections do not improve parameter recovery, and there is little aggregate
forgetting for them to fix.** What improves recovery is pricing accuracy.

| arm | params | final recovery RMSE | best → final regret | forget rate | final IV error | loss-of-fit events |
|---|---:|---:|---:|---:|---:|---:|
| A plain 5×160 (B0 architecture) | 107k | 0.280 ± 0.044 | +0.006 | 0.24 | 0.000401 | 1.3 |
| B plain 11×160 | 262k | **0.194 ± 0.038** | +0.001 | 0.23 | **0.000323** | 5.0 |
| C 5×160 + 3 residual blocks | 262k | 0.254 ± 0.017 | +0.003 | 0.21 | 0.000393 | 1.3 |

Paired by seed:
- **C vs B (same size; only difference is the skip connection):** recovery worse by 0.060, C better on 1/3
  seeds; IV error worse on 3/3; forget rate lower by 0.02 (2/3); far fewer loss-of-fit events (3/3).
- **B vs A:** recovery better by 0.086 on **3/3** seeds; IV error better on 3/3.
- **C vs A:** recovery better by 0.026 on 2/3 seeds.

## Findings

1. **The apparent forgetting was mostly the 4-case check.** Best-to-final regret is ≤ 0.015 in every
   run on 24 truths. Resampling 4 of 24 cases, 93% of 4-case checks show a "rise" on a run that improves
   monotonically.
2. **Per-estimate churn is real but architecture-independent.** About 21–24% of (case, parameter) pairs
   recovered at one checkpoint are lost by the next, in every arm. It comes from directions the price
   surface barely constrains, not from the network.
3. **Pricing accuracy drives recovery.** Across all 9 runs, final IV error vs final recovery RMSE:
   Pearson r = +0.86 (p = 0.003), Spearman = +0.88 (p = 0.002). The identifiability analysis predicts
   this: a lower pricing error shrinks the flat zone in proportion.
4. **Residual connections stabilise training but do not price better.** C loses its fit less often than
   B, but ends 22% less accurate and close to A. With zero-initialised outputs scaled by 0.1, the extra
   capacity is not turning into accuracy at LR 1e-3. That is a plausible cause, not a tested one.
5. **Unrecovered parameters are the same in every arm:** slow-factor vol-of-vol and speed, and the
   slow/fast split of θ and v0. The totals θ_s+θ_f (76–88% within 5%) and v0_s+v0_f (100%) are recovered.
   All ten within tolerance at once: 0–3% of cases.

## Checkpoint selection (affects the frozen runbook)

The trainer's 4-case selection picked a checkpoint worse than the 24-case best in 5 of 9 runs:
A17 by **55%** (this is B0), B17 by 74%, B43 by 20%, B29 by 16%, C43 by 5%. **B0's saved
`model.safetensors` is step 6000, 0.372 on 24 truths, against 0.239 at step 10000.** B1 and B2 start
from that checkpoint. Recorded only: the runbook forbids changes after observing B0, so any change is
a decision for the owner or mentor.

## Caveats

Three seeds per arm; one dataset; synthetic truths on a 21-strike × 6-maturity grid. B vs C also differs
in zero-initialisation and the 0.1 output scale, which are inherent to this residual implementation.
Exploratory: nothing here modifies or re-selects B0, B1 or B2.
