# Proposed parameter comparison

This development study screens seven fixed synthetic truths: the published DH1
control and six explicitly study-defined sensitivity proposals. The latter
must not be described as published calibrations or NSE market parameters.
The source and numerical reproduction of DH1 are documented in
[the fixed-truth pilot](LITERATURE_FIXED_TRUTH_PILOT.md).

## Changes tested

Every row starts from DH1 unless stated otherwise. The complete ten numbers
are frozen in `configs/literature_parameter_sweep.json`.

| Candidate | Difference from published DH1 |
|---|---|
| Published DH1 | Unchanged control |
| Separated speeds | kappa_s=.3, kappa_f=3, sigma_s=.18 (retains Feller) |
| Rapid fast factor | kappa_f=6 |
| Asymmetric correlation | rho_s=-.8, rho_f=-.2 |
| Mixed correlation | rho_s=-.7, rho_f=+.2 |
| Balanced initial variance | v0_s=v0_f=.28; same total initial variance |
| Lower initial variance | v0_s=v0_f=.15, sigma_s=.18 |

These values were specified before observing this sweep's results. They are
mechanism-based sensitivity choices, not an exhaustive search, random draws,
or proposals set equal to a previous run's fitted answer. The existing solver,
decoder, loss, data domain, 246 fitting quotes, 240 interleaved validation quotes,
18,000 LHS collocation points, 1,000 Adam steps and 100 L-BFGS iterations are
unchanged across candidates. No candidate-specific tuning is performed.

## Selection and confirmation

1. Fit all seven truths from seeds 17 and 43, reporting all fourteen outcomes.
2. Rank first by the number of complete ten-parameter recoveries, then by the
   median across seeds of the maximum tolerance-normalized parameter error.
   The latter is the worst parameter's error divided by its allowed error;
   a value <=1 means every parameter passes for that fit. Break ties using
   worst repriced validation IV RMSE, then case ID.
3. Save the selected truth and full ranking **before** confirmation. Do not
   switch candidates after observing confirmation outcomes.
4. Refit the selected truth, plus the published control, with seeds 89 and 97.
   Test the selected truth at 160 additional coordinates with maturities
   45,120,270,540 days and S/K from .5125 to 1.4875. These maturities were not
   used in training or screening. No updates follow that evaluation.
5. Require every selected-candidate fit to pass all ten parameter gates,
   repriced IV RMSE <=.005, fresh scaled PDE RMSE <=.01, zero sampled
   negative-convexity points, and the fresh quote repricing gate. Otherwise
   `final_implementation_parameters` remains null.

Parameter gates remain 5% relative error for eight positive quantities and
.05 absolute error for each correlation. The new price/PDE promotion gates
are study-defined screening requirements, not literature-prescribed thresholds.

## Honest use in the final implementation

The screening data formerly named `holdout.npz` in the reused pilot code now
serve as **validation** for benchmark selection. They are not an untouched
final test. Truth errors are used only for post-fit development selection,
not as labels in any optimization objective. This is a disclosed selection
procedure, not a claim that ground truth is invisible to the researcher.

Even if the selected candidate passes, it would qualify only as an explicitly
selected synthetic demonstration case. Fresh seeds and quotes do not establish
generalization to new parameter sets, or superiority over Single Heston. Final
research reporting must retain the entire sweep, including failed cases.
Selecting an easy truth cannot repair a generally unreliable calibration method.

Run from the controlled repository with the project environment:

```sh
python scripts/mentor_dh_pinn/run_literature_parameter_sweep.py --out outputs/NEW_UNIQUE_SWEEP_DIRECTORY
python -m pytest -q tests/test_literature_parameter_selection.py tests/test_literature_fixed_truth_pilot.py tests/test_regular_pinn_torch_physics.py tests/test_torch_pricer.py
```

The runner uses three local training processes and preserves every process log,
checkpoint, initial guess, protocol, data hash, ranking and confirmation outcome.
No previous files are overwritten. Frozen V2 and final implementation defaults
are left untouched unless a candidate genuinely qualifies for promotion.
