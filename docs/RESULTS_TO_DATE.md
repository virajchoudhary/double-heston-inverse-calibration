# Results to Date

Status date: 06 August 2026

## 1. Verified engineering results

| Check | Fresh result |
|---|---|
| Python | 3.13.9 |
| PyTorch | 2.11.0+cpu |
| Compute device | CPU; CUDA unavailable |
| Compilation | Passed |
| Automated tests | 18 passed |
| Test runtime | 4.89 seconds |
| Smoke test | Passed |
| Smoke surfaces | 48 |
| Surface split | 33 train / 7 validation / 8 test |
| ANN input size | 108 |
| ANN output size | 10 |
| Test prediction shape | `(8, 10)` |
| Checkpoint creation | Passed; reproducible artifact excluded from Git |
| Model selection | Validation only; test data not used |
| PDE loss | Absent (`false`) |
| Smoke data status | `NOT_RESEARCH_DATA` |
| Cross-process repeatability | Two consecutive histories and summaries had identical SHA-256 hashes |
| Repricing evaluation | Blocked: validated Double Heston engine absent |

The smoke flow verifies deterministic surface generation, surface-level splitting, tensor shapes, parameter-supervised loss computation, target standardization, validation checkpointing, prediction generation, and output serialization.

## 2. Results that do not yet exist

- Genuine Double Heston synthetic parameter-recovery results
- ANN results trained on validated Double Heston surfaces
- Repricing results from ANN-predicted parameters
- Noise-robustness results at 0%, 0.5%, 1%, and 2%
- Five-seed stability results
- Chronological NIFTY EOD validation
- ANN versus PINN versus numerical calibration versus Standard Heston results

Current blockers are the frozen validated pricing implementation, all imported helpers, controlled recovery tests, confirmed parameter bounds and provenance, and the final NIFTY EOD surface contract.

## 3. Research claims that must not be made

- Do not report dummy parameter metrics as financial-model performance.
- Do not call the dummy mapping a Double Heston pricer.
- Do not claim parameter-recovery accuracy, pricing accuracy, robustness, generalization, or superiority from the smoke test.
- Do not describe historical calibrated parameters as unique truth or ANN ground truth.
- Do not describe the ANN as research-trained.
