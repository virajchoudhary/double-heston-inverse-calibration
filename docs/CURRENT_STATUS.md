# Current Project Status

Status date: 10 August 2026

## Approved research objective

Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

## Current completed milestone

The canonical Double Heston pricing/calibration foundation, independent pricing verification, reviewed synthetic-sampling foundation, ANN infrastructure, and deterministic official-NSE Stage A candidate-selection milestone are complete. The selected primaries are NTPC, CIPLA, INFY, and HDFCBANK; POWERGRID, SUNPHARMA, TCS, and ICICIBANK are retained as backups. The original three-date Power comparison was unresolved, and the predeclared five-Wednesday July extension selected NTPC at moderate confidence. Bloomberg was not used. The next research milestone is common-support analysis across the four primaries, followed by the G2 representation decision.

The unavailable teammate engine is being replaced by an independently implemented canonical Double Heston engine. Equivalence to the unavailable source is not claimed.

| Component | Status |
|---|---|
| ANN project structure | Complete |
| Fixed ten-parameter contract | Complete |
| Canonical pricing engine | 36-case independent benchmark passed; freeze evidence created |
| Engine-focused benchmark set | 36 / 36 passed |
| Full automated suite | 161 passed at the latest validated milestone |
| Independent reference integrations | 36 / 36 reliable; zero warnings or failures |
| Production/reference agreement | All 64-node and 96-node cases passed |
| ANN research pricing adapter | Integrated |
| Clean controlled recovery | Completed; one exact best result, start sensitivity retained |
| One-percent noise recovery | Completed; parameter instability and boundary-near solutions observed |
| Genuine-engine pilot | 12 surfaces / 1,296 quotes |
| Prior provisional-bounds audit | 5,000 raw / 2,776 accepted / 2,224 rejected |
| Historical priced bounds-audit subset | 250 surfaces / 21,000 prices; all validity checks passed |
| Reviewed sampling audit | 19,000 candidates; interior 8,116 accepted, wide-valid 3,371 accepted |
| Normal synthetic core readiness | `CORE_DATASET_READY = true` under the existing candidate surface contract |
| Challenge stress readiness | `CHALLENGE_STRESS_READY = false`; retained stress cases remain separate |
| Stage A NSE screen | `COMPLETE`; 24 candidate stock surfaces across three dates |
| Primary Stage A source | Official NSE CM and F&O UDiFF bhavcopies |
| Candidate selection | `COMPLETE`; NTPC, CIPLA, INFY, and HDFCBANK |
| Surface-representation G2 gate | `NOT_PASSED` |
| Current 108-input grid | `REJECTED_AS_FINAL_UNCHANGED_GRID` |
| Replacement representation | `OPEN`; no feature count frozen |
| Final 10,000-surface research dataset | Not generated |
| Development smoke test | Passing; remains `NOT_RESEARCH_DATA` |
| Full ANN research training | Not started |
| PINN development/comparison | Not started |
| Frozen real-market evaluation | Not started |

## Honest research boundary

The independent benchmark materially strengthens the pricing evidence, but agreement between two implementations is not proof of universal correctness. The historical 5,000-row uniform bounds audit did not approve that earlier sampling design: 44.48% of raw candidates were rejected, and among the 2,776 accepted vectors 32.6729% were near at least one declared boundary and 7.0605% were near a Feller boundary. Rejected vectors are reported separately and are not counted as boundary-near.

The reviewed four-distribution sampler separates normal training populations from deliberately difficult boundary-challenge and OOD populations. The normal clean core is ready under the current candidate surface contract, while retained challenge stress cases remain a separate non-ready evidence set. This does not establish externally valid parameter ranges, unique parameter recovery, ANN/PINN performance, or market generalization.

The 1% controlled noise experiment remains an important research result: comparatively reasonable price fit can coexist with severe deterioration in ten-parameter identification. This motivates the later ANN and physics-informed inverse comparisons rather than constituting evidence that either neural method is superior.

## Reviewed sampling follow-up

The reviewed sampler separates interior, wide-valid, boundary-challenge, and OOD populations. Interior acceptance is 81.16% and wide-valid acceptance is 67.42%; challenge rows are balanced across four labels and OOD rows lie outside normal `kappa_fast` support with no train/validation assignment.

Four deliberately difficult challenge surfaces produced tiny 64-node numerical-tolerance stress failures. They were retained as evidence, pass at 96 Gauss-Laguerre nodes, agree with the independent adaptive reference within the frozen comparison tolerance, and remain excluded from ordinary ANN training. Consequently normal core readiness and challenge stress readiness are tracked separately rather than forcing one global ready/not-ready label. See [REVIEWED_PARAMETER_SAMPLING.md](REVIEWED_PARAMETER_SAMPLING.md).

## Completed Stage A NSE market-support screen

The prepared synthetic plan still uses the candidate 108-input contract:

- 9 log-moneyness points;
- 6 candidate maturities: 7, 14, 30, 60, 90, and 180 days;
- calls and puts;
- 10 ordered Double Heston targets.

The mentor-updated v2 execution plan requires real-market support evidence before the final 10,000-surface generation. The official-NSE screen and candidate-selection milestone have supplied the Stage A evidence, but common-support analysis and the replacement representation remain open.

Stage A processed:

- Power candidates: NTPC and POWERGRID;
- Healthcare/Pharma candidates: SUNPHARMA and CIPLA;
- IT candidates: INFY and TCS;
- Financial/Banking candidates: ICICIBANK and HDFCBANK;
- NIFTY as a separate non-ranked reference;
- audit dates 01 July, 15 July, and 22 July 2026;
- one surface as one underlying-date containing near, mid, and far expiry slices together;
- price-usability checks separate from volume/open-interest activity;
- futures-implied carry inputs audited for availability without calculating or selecting final carry.

All eight stock candidates were present on all three dates. The result comprises 24 stock surfaces and 4,740 stock-option rows. Actual stock-option DTE patterns were `27/55/90`, `13/41/76`, and `6/34/69` by date. All 24 CM closes equaled the corresponding unique F&O `UndrlygPric`; all stock futures expiries aligned with stock-option expiries; and no selected stock row had an `XpryDt` / `FininstrmActlXpryDt` mismatch.

The current grid is not supportable unchanged as the final representation. The 180-day node was outside observed expiry support for all 24 surfaces, while the `-0.30` and `+0.30` log-moneyness wings were observed in only 2/72 and 7/72 stock-expiry slices. The central `-0.10` through `+0.10` nodes were observed in all 72 slices. This evidence rejects the final unchanged 108-grid but does not freeze a replacement. `G2 = NOT_PASSED`.

Official NSE is the primary Stage A source. Free NSE bhavcopy lacks historical bid/ask and quote-size fields, and Bloomberg was not used. The original three-date Power comparison remained unresolved; the separate five-Wednesday July extension selected NTPC at moderate confidence. The selected primaries are NTPC, CIPLA, INFY, and HDFCBANK. See [STAGE_A_CANDIDATE_SELECTION.md](STAGE_A_CANDIDATE_SELECTION.md).

## Generation boundary

`CORE_DATASET_READY = true` means the reviewed normal parameter population and leakage/split logic are ready **under the existing candidate surface contract**. It does not authorize immediate final dataset generation under the mentor-updated pipeline.

The final 10,000-surface run remains blocked until common-support analysis across the four primaries leads to a G2-approved representation and the synthetic surface contract is updated and revalidated. No final research dataset, ANN research training, PINN derivation or training, or frozen real-market evaluation has occurred.

```text
STAGE_A_NSE_SCREEN = COMPLETE
STAGE_A_CANDIDATE_SELECTION = COMPLETE
POWER_SELECTION = NTPC
SELECTED_PRIMARY_SET = NTPC | CIPLA | INFY | HDFCBANK
BACKUP_SET = POWERGRID | SUNPHARMA | TCS | ICICIBANK
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
REPLACEMENT_REPRESENTATION = OPEN
G2 = NOT_PASSED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
```
