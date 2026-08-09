# Current Project Status

Status date: 09 August 2026

## Approved research objective

Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

## Current completed milestone

The canonical Double Heston pricing/calibration foundation, independent pricing verification, reviewed synthetic-sampling foundation, ANN infrastructure, and Stage A real-market availability-audit scaffold are complete. The next execution milestone is collection of small Stage A Bloomberg availability snapshots before the final market representation is frozen.

The unavailable teammate engine is being replaced by an independently implemented canonical Double Heston engine. Equivalence to the unavailable source is not claimed.

| Component | Status |
|---|---|
| ANN project structure | Complete |
| Fixed ten-parameter contract | Complete |
| Canonical pricing engine | 36-case independent benchmark passed; freeze evidence created |
| Engine-focused benchmark set | 36 / 36 passed |
| Full automated suite | 96 passed at the latest validated milestone |
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
| Stage A availability-audit scaffold | Complete and validated; no Bloomberg observations collected yet |
| Market-data G1 gate | Not passed |
| Surface-representation G2 gate | Not passed; representation remains `PROVISIONAL` / `OPEN` |
| Current 108-input grid | Candidate contract only; not frozen for final research generation |
| Final 10,000-surface research dataset | Not generated |
| Development smoke test | Passing; remains `NOT_RESEARCH_DATA` |
| Full ANN research training | Not started |
| PINN development/comparison | Not started |
| Real sector/NIFTY validation | Not started |

## Honest research boundary

The independent benchmark materially strengthens the pricing evidence, but agreement between two implementations is not proof of universal correctness. The historical 5,000-row uniform bounds audit did not approve that earlier sampling design: 44.48% of raw candidates were rejected, and among the 2,776 accepted vectors 32.6729% were near at least one declared boundary and 7.0605% were near a Feller boundary. Rejected vectors are reported separately and are not counted as boundary-near.

The reviewed four-distribution sampler separates normal training populations from deliberately difficult boundary-challenge and OOD populations. The normal clean core is ready under the current candidate surface contract, while retained challenge stress cases remain a separate non-ready evidence set. This does not establish externally valid parameter ranges, unique parameter recovery, ANN/PINN performance, or market generalization.

The 1% controlled noise experiment remains an important research result: comparatively reasonable price fit can coexist with severe deterioration in ten-parameter identification. This motivates the later ANN and physics-informed inverse comparisons rather than constituting evidence that either neural method is superior.

## Reviewed sampling follow-up

The reviewed sampler separates interior, wide-valid, boundary-challenge, and OOD populations. Interior acceptance is 81.16% and wide-valid acceptance is 67.42%; challenge rows are balanced across four labels and OOD rows lie outside normal `kappa_fast` support with no train/validation assignment.

Four deliberately difficult challenge surfaces produced tiny 64-node numerical-tolerance stress failures. They were retained as evidence, agree under the higher-accuracy/reference checks documented by the project, and remain excluded from ordinary ANN training. Consequently normal core readiness and challenge stress readiness are tracked separately rather than forcing one global ready/not-ready label. See [REVIEWED_PARAMETER_SAMPLING.md](REVIEWED_PARAMETER_SAMPLING.md).

## Market-support representation gate

The prepared synthetic plan still uses the candidate 108-input contract:

- 9 log-moneyness points;
- 6 candidate maturities: 7, 14, 30, 60, 90, and 180 days;
- calls and puts;
- 10 ordered Double Heston targets.

However, the mentor-updated v2 execution plan requires a real-market availability/coverage audit before that representation is used for final 10,000-surface generation. Individual-stock option maturity support may not justify the existing six-tenor grid, so the current representation is explicitly `PROVISIONAL` and `OPEN`.

Stage A currently specifies:

- Power candidates: NTPC and POWERGRID;
- Healthcare/Pharma candidates: SUNPHARMA and CIPLA;
- IT candidates: INFY and TCS;
- Financial/Banking candidates: ICICIBANK and HDFCBANK;
- NIFTY as a separate non-ranked reference;
- audit dates 01 July, 15 July, and 22 July 2026;
- one surface as one underlying-date containing near, mid, and far expiry slices together;
- price-usability checks separate from volume/open-interest activity;
- futures-implied carry audited for availability but not selected as the final carry convention.

The Stage A code/configuration/documentation scaffold is complete and tested, but no Bloomberg observations have been collected. Therefore neither G1 (market-data gate) nor G2 (representation gate) has passed. No 54-, 57-, or other replacement representation has been frozen.

## Generation boundary

`CORE_DATASET_READY = true` means the reviewed normal parameter population and leakage/split logic are ready **under the existing candidate surface contract**. It does not authorize immediate final dataset generation under the mentor-updated pipeline.

The final 10,000-surface run remains blocked until Stage A evidence establishes common maturity/moneyness support, G2 freezes the representation, and any required configuration/tests are updated before generation. No final research dataset, ANN research training, PINN training, or real-market validation has occurred.
