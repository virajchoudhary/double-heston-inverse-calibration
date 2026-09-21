# Fixed research-based truth pilot: results, 15 September 2026

**Outcome: training completed, but reliable parameter recovery was NOT achieved.**
All six prescribed fits finished. Zero of six passed all ten parameter gates;
only one of 60 individual parameter checks passed. Do not promote this pilot
as a calibrated model or overwrite the previous research model with it.

## What was actually tested

One exactly published Double Heston parameter set (DH1), plus two explicitly
study-defined sensitivity variants of that set, each from two blind starts.
The derived cases must not be described as additional published parameter sets.
Source: [Kyriakou, Brignone and Fusai, DOI 10.1287/opre.2022.2422](https://doi.org/10.1287/opre.2022.2422),
Tables 2 and 3 of the accepted manuscript. The teacher reproduces the paper's
DH1 call value 26.9504 after rounding.

The exact values and full experimental method are in
[LITERATURE_FIXED_TRUTH_PILOT.md](LITERATURE_FIXED_TRUTH_PILOT.md).
The experiment used 246 fitting quotes, 240 held-out strike-interpolation quotes,
six maturities (30,60,90,180,365,730 days), and an 18,000-point LHS collocation pool
per fit. This was a fixed-budget pilot: 1,000 Adam steps followed by at most 100
L-BFGS iterations, not a run demonstrated to have converged.

## All results, including failures

IV errors below are in **absolute annualized volatility units**: .01 means one
volatility percentage point. Neural IV uses the network's output. Repriced IV
uses the recovered ten parameters in the independent Fourier pricing engine,
without any parameter refinement or optimizer calls to that engine.

| Truth case | Seed | Parameters passing /10 | Fitting neural IV RMSE | Held-out neural IV RMSE | Held-out repriced IV RMSE | Fresh scaled PDE RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Published DH1 | 17 | 0 | .002778 | .002720 | .299665 | .219874 |
| Published DH1 | 43 | 0 | .004936 | .004776 | .013224 | .022427 |
| Derived faster factor | 17 | 0 | .003649 | .003562 | .306484 | .107761 |
| Derived faster factor | 43 | 0 | .005042 | .004883 | .018589 | .018965 |
| Derived lower vol-of-variance | 17 | 0 | .002401 | .002323 | .298716 | .112483 |
| Derived lower vol-of-variance | 43 | 1 | .003797 | .003740 | .016920 | .019705 |

Parameter gates were unchanged: eight positive quantities within 5% relative
error, and two correlations within .05 absolute error. Reporting all seeds
avoids selecting the seed closest to the known answer.

## Published DH1: every parameter

| Parameter | Fixed true value | Recovered, seed 17 | Recovered, seed 43 |
|---|---:|---:|---:|
| kappa_s | .9 | .267249 | .226250 |
| theta_s | .1 | .063759 | .080286 |
| sigma_s | .36 | .063365 | .060633 |
| rho_s | -.5 | -.285636 | -.295569 |
| v0_s | .36 | .059561 | .145831 |
| kappa_f | 1.2 | 2.933512 | 2.389243 |
| theta_f | .15 | .077116 | .196725 |
| sigma_f | .2 | .291990 | .422557 |
| rho_f | -.5 | -.073490 | -.198597 |
| v0_f | .2 | .108667 | .418404 |

## Interpretation

Hard-coded research values give us an auditable answer against which to test;
they do not make the inverse problem automatically identifiable or easy to solve.
The discrepancy between neural and independently repriced errors shows that
the network's ability to fit quotes is not yet tied tightly enough to correct
structural parameters. The large fresh PDE residuals also prevent a claim of
an accurately solved pricing PDE. This is consistent with an under-resolved or
under-optimized joint inverse solver, but this pilot does not isolate the cause.

Small train/held-out neural-error differences concern only interleaved strikes
on the same synthetic surfaces. They do not establish absence of overfitting,
unseen-case recovery, real-market validity, or superiority to Single Heston.
Price and IV targets are transformations of the same quotes, not extra
independent information that would eliminate non-identifiability.

## Checks that passed, and what they do not prove

- Ten automated software tests passed, covering teacher pricing, full PDE
  derivatives, gradient flow, constraints, payoff, and a static guard against
  truth/holdout/pricer access in the training function.
- All source and data hashes recorded at generation remained unchanged.
- All three training truths satisfy strict Feller; published DH2 remains
  excluded unchanged because it does not satisfy this experiment's constraint.
- Initial guesses for each seed were identical across cases, not truth-centered.
- No training/held-out coordinate overlap; no target-parameter columns in the
  fitting files; all synthetic labels were finite with valid IV inversions.
- 96- versus 128-node teacher prices agreed within 3e-14 normalized price.
- 4,096 fresh LHS points per final fit produced finite PDE diagnostics and no
  sampled negative-convexity violations. That finite sample is not a global
  no-arbitrage guarantee. PDE accuracy itself was not satisfactory.

These checks support reproducibility and disclose the failure. They do not
convert the failed calibration into successful research evidence or provide a
universal guarantee against leakage in future modifications.

## Files and continuation

All generated artifacts are under `outputs/literature_fixed_truth_pilot_20260915/`:

- `truth_and_sources.json`, `protocol.json`, `data_audit.json`;
- `collocation.npz` and per-case `train.npz` / `holdout.npz`;
- per-case/seed `initial_parameters.json`, `training.jsonl`, `checkpoint.pt`, `fit.json`;
- `results.json`: all ten true/fitted values, errors and gates for every fit;
- `postfit_audit.json`: fresh PDE checks and checkpoint hashes.

Training code: `scripts/mentor_dh_pinn/literature_fixed_truth_pilot.py`.
Post-fit audit: `scripts/mentor_dh_pinn/audit_literature_pilot.py`.

Preserve this failed pilot. A next development experiment should investigate
PDE/boundary accuracy and optimization convergence first, with a separately
declared budget and unchanged recovery gates. Do not tune against these known
answers and then call them untouched tests. A stronger claim would require
additional independent benchmark truths and blind final evaluation. Longer
training is a hypothesis to test, not a guarantee of ten-parameter recovery.

No frozen V2 B0/B1/B2 settings, original market data, Kaggle data, or prior
checkpoints were changed by this pilot. These outputs have not been pushed to GitHub.
