# Proposed parameter sweep: completed results

**Decision: no candidate qualifies for the final calibration implementation.**
Seven fixed truths were screened with two starting seeds each; the selected
candidate and published control were then fitted with two fresh seeds. All
18 fits completed. Zero of 18 recovered all ten parameters; only two of 180
individual parameter checks passed. The final parameter setting remains unset
in the decision artifact; no prior implementation was replaced.

## Best-ranked development candidate, not a validated answer

The predeclared parameter-error ranking selected `proposed_separated_speeds`.
These are **synthetic true values used to generate prices**, not values the
PINN successfully recovered and not an estimated market calibration.

| Parameter | Slow factor | Fast factor |
|---|---:|---:|
| kappa | .3 | 3.0 |
| theta | .1 | .15 |
| sigma | .18 | .2 |
| rho | -.5 | -.5 |
| Initial variance v0 | .36 | .2 |

This is a study-defined sensitivity variation around published DH1, not a
second published parameter set. Its Feller margins are .0276 and .86, and all
ten values are inside the unchanged pilot decoder's admissible domain.

## Full screening ranking

The ranking prioritizes complete recovery counts, then the median across seeds
of each fit's worst tolerance-normalized parameter error. Values <=1 would
mean the worst parameter meets its allowed error for that fit. All candidates
had zero complete recoveries; the ranking therefore identifies only the
least-bad observed case. Price accuracy alone was not the selection criterion.

| Rank | Candidate | Complete recoveries /2 | Median worst error / tolerance | Worst validation repriced IV RMSE |
|---|---|---:|---:|---:|
| 1 | Separated speeds | 0 | 18.5981 | .332486 |
| 2 | Rapid fast factor | 0 | 22.7088 | .293613 |
| 3 | Published DH1 control | 0 | 25.5738 | .299665 |
| 4 | Lower initial variance | 0 | 27.2930 | .016589 |
| 5 | Asymmetric correlation | 0 | 27.4958 | .297791 |
| 6 | Balanced initial variance | 0 | 28.0062 | .295483 |
| 7 | Mixed correlation | 0 | 28.5081 | .301165 |

Lower initial variance had the smallest worst repriced IV error, but did not
recover the true parameters. It was not substituted for the selected candidate
after seeing the confirmation results.

## Selected candidate: independent confirmation

The selected truth and ranking were saved before the following two seeds ran.
Both used the same training quotes and budget as screening. The final quote
check uses 160 coordinates at 45,120,270,540 days, maturities excluded from
training and screening. No optimization followed this check.

| Fresh seed | Parameter checks passed /10 | Neural IV RMSE, unseen quotes | Repriced IV RMSE, unseen quotes | Fresh scaled PDE RMSE |
|---|---:|---:|---:|---:|
| 89 | 1 | .003100 | .099611 | .043054 |
| 97 | 0 | .002036 | .299535 | .072266 |

The required repriced IV RMSE is <=.005 and the required scaled PDE RMSE is
<=.01. IV errors are absolute annualized volatility units: .01 is one
volatility percentage point. All ten parameter checks must pass in every seed;
eight quantities use 5% relative error, and correlations use .05 absolute error.
The candidate fails parameter, repricing and PDE gates.

## Interpretation and limitations

The network can approximate the price/IV surface while compensating for wrong
parameters. Repricing the recovered values independently exposes that problem.
The source prices and admissible parameter values are not the main evidence
of failure here; the failure is that this joint PINN does not recover them
reliably under the tested budget. This does not prove recovery impossible or
prove that longer training would fix it.

Selecting a truth because it is easier for the current model is benchmark
selection. It must be disclosed and must not be described as universal recovery,
market calibration, or superiority over Single Heston. Screening `holdout.npz`
was used as validation for selection; only fresh confirmation quotes were
unseen at selection time. Fresh seeds and quotes still share the selected
structural truth, so they are not tests of new parameter-set generalization.

## Reproducibility checks

- All seven truths passed Feller, admissible-domain, finite-price and IV checks.
- Fourier teacher prices agreed between 96 and 128 nodes within about 3e-14.
- The published control reproduced both earlier seed fits exactly.
- Source/data hashes and seed-specific initialization equality passed.
- All 18 fits received the same architecture and 1,000 Adam + up to 100
  L-BFGS iteration budget, with an 18,000-point collocation pool.
- Fresh PDE diagnostics used 4,096 points per fit; no sampled negative convexity
  was found, but PDE error remained too large. This is not a global guarantee.
- Twelve automated software tests passed. Software correctness checks do not
  imply scientifically successful parameter recovery.

## Saved files and final-use decision

Protocol: [PROPOSED_PARAMETER_SELECTION.md](PROPOSED_PARAMETER_SELECTION.md).
Complete proposed values: `configs/literature_parameter_sweep.json`.
Runner: `scripts/mentor_dh_pinn/run_literature_parameter_sweep.py`.

All outputs are in `outputs/literature_parameter_sweep_20260915/`, including
the frozen selection protocol, source hashes, full screening ranking, all
18 checkpoints and process logs, every true/recovered parameter and error,
per-stage audits, and fresh quote results.

`decision.json` records `eligible_for_final_implementation: false` and
`final_implementation_parameters: null`. The best-ranked case is retained for
development only. Frozen V2, previous experiment outputs, market data, Kaggle,
GitHub and final model defaults were not changed.

The appropriate next development step is to improve and validate the PINN's
physics/optimization coupling, rather than choose a more convenient answer.
Keep this full failed sweep when reporting later improvements; assess a final
method on additional predeclared parameter truths not used to tune it.
