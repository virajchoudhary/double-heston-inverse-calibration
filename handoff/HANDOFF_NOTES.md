# Takeover Notes

## Status on 06 August 2026

`Heston_Double_Heston_Validated_Teammate_Handoff_FINAL.pdf` was the only teammate handoff file available at takeover time. A faithful Markdown technical companion was generated as `HESTON_DOUBLE_HESTON_TEAM_CONTEXT.md`.

The existing Single Heston and Double Heston work is frozen as a validated prototype with disclosed limitations. Its historical results are based on 11 power-sector stocks. The final approved capstone scope is NIFTY end-of-day European option surfaces under the canonical Double Heston model.

Historical Double Heston parameter tables are not supervised ANN ground truth. ANN labels must come from parameter vectors known before synthetic prices are generated. No ANN or PINN research result exists yet, and the smoke-test outputs created by this project are explicitly `NOT_RESEARCH_DATA`.

## Completed today

- Audited the accessible workspace for Heston source, synthetic recovery, ANN/PINN code, requirements, bounds, Git state, and project documents.
- Copied the validated PDF into `handoff/` without removing or changing the original.
- Created the Markdown companion and this takeover note.
- Locked the exact ten-parameter order and strict positivity, slow/fast, Feller, and joint-correlation constraints.
- Created a deterministic call/put surface grid with 108 normalized-price inputs.
- Created the ordinary PyTorch ANN, bounded transform, and training-only target standardization.
- Created research-mode and smoke-test-mode dataset paths with no silent fallback.
- Added supervised training, parameter evaluation, repricing evaluation, and an end-to-end CPU smoke test.
- Added unit tests and configuration templates.

## Blocked research work

Real Double Heston synthetic generation and ANN repricing evaluation are blocked. The project intentionally raises `MissingPricingEngineError` instead of pretending that its development-only mapping is a financial pricer. Final parameter bounds are also blocked and remain null in `configs/parameter_bounds_TEMPLATE.yaml`.

## Exact files needed from the teammate

1. Frozen `double_heston.py` used for the validated report.
2. Frozen `test_double_heston.py` and controlled synthetic-recovery test/data.
3. Any local modules imported by `double_heston.py`, including characteristic-function, quadrature, option-bound, or IV helpers.
4. A short invocation example for calls and puts, including parameter order and array shapes.
5. The confirmed ten-parameter bounds with provenance and version.
6. Numerical settings used by the validated engine, including integration nodes/tolerances and failure behavior.
7. Selected checksum manifest for the frozen source.
8. Final NIFTY option-surface data contract: EOD timestamp, exact expiries, rate/dividend inputs, quote filters, interpolation or masking policy, and chronological train/validation/test dates.

For historical reproducibility only, the referenced `single_heston.py`, forecast/comparison/audit scripts, sanitized catalogs, and failure files are also needed. They must remain frozen and must not be used to manufacture ANN labels.

## Rerun commands

Run from the parent directory that contains `ann_inverse_calibration`:

```powershell
python -m compileall ann_inverse_calibration
python -m pytest ann_inverse_calibration/tests -q
python -m ann_inverse_calibration.src.run_smoke_test
```

The smoke test is infrastructure validation only. Its data and metrics are not Double Heston research evidence.
