# Current Project Status

Status date: 07 August 2026

## Approved research objective

Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

## Current completed milestone

Independent canonical Double Heston pricing, adaptive-quadrature benchmarking, provisional-bounds audit, controlled synthetic validation, and ANN adapter integration.

The unavailable teammate engine is being replaced by an independently implemented canonical Double Heston engine. Equivalence to the unavailable source is not claimed.

| Component | Status |
|---|---|
| ANN project structure | Complete |
| Fixed ten-parameter contract | Complete |
| Canonical pricing engine | 36-case independent benchmark passed; freeze evidence created |
| Engine-focused tests | 36 passed |
| Full automated suite | 84 passed in the current post-correction run |
| Independent reference integrations | 36 / 36 reliable; zero warnings or failures |
| Production/reference agreement | All 64-node and 96-node cases passed |
| ANN research pricing adapter | Integrated |
| Clean controlled recovery | Completed; one exact best result, start sensitivity retained |
| One-percent noise recovery | Completed; parameter instability and boundaries observed |
| Genuine-engine pilot | 12 surfaces / 1,296 quotes |
| Prior provisional-bounds audit | 5,000 raw / 2,776 accepted / 2,224 rejected |
| Historical priced bounds-audit subset | 250 surfaces / 21,000 prices; all validity checks passed |
| Sampling configuration | Reviewed structure created; numerical ranges still provisional |
| Reviewed sampling audit | 19,000 candidates; interior 8,116 accepted, wide 3,371 accepted |
| Freeze decision | `NEEDS_SAMPLER_CORRECTION` |
| Development smoke test | Passing; remains `NOT_RESEARCH_DATA` |
| Full ANN research training | Not started |
| NIFTY validation | Not started |
| PINN development/comparison | Not started |

## Honest research boundary

The independent benchmark materially strengthens the pricing evidence, but agreement between two implementations is not proof of universal correctness. The historical 5,000-row uniform bounds audit did not approve that earlier sampling design: 44.48% of raw candidates were rejected, and among the 2,776 accepted vectors 32.6729% were near at least one declared boundary and 7.0605% were near a Feller boundary. Rejected vectors are reported separately and are not counted as boundary-near. The reviewed four-distribution sampler is described below and remains non-ready for a different reason: four retained challenge pricing failures. The work does not establish equivalence to the missing implementation, unique parameter recovery, externally valid bounds, ANN performance, or market generalization. Historical power-sector calibrations remain reproducibility artifacts and are not ANN truth labels.

## Reviewed sampling follow-up

The reviewed sampler separates interior, wide-valid, boundary-challenge, and OOD
populations. Interior acceptance is 81.16% and wide-valid acceptance is 67.42%;
challenge rows are balanced across four labels and OOD rows are outside the
normal `kappa_fast` support with no train/validation assignments. The 1,250 clean
priced surfaces retain four challenge pricing failures, so the current gate is
`NEEDS_SAMPLER_CORRECTION`. The complete evidence is in
[REVIEWED_PARAMETER_SAMPLING.md](REVIEWED_PARAMETER_SAMPLING.md).
