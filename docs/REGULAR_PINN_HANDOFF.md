# Regular Single/Double Heston PINN — continuation context

## Outcome and scope

The regular Single Heston pricing-PINN workflow has been extended to Double
Heston and actually trained/tested. Accurate, reliable ten-parameter recovery
has **not** been achieved. Do not claim perfect calibration, universal
identification, no possible leakage, or a completed three-PINN comparison.
All new data in this iteration is explicitly synthetic; original NSE records,
PostgreSQL and Kaggle datasets were not changed.

Eight development runs completed: 112,000 optimizer updates in total, excluding
the 200-step runtime pilot. All start from training seed 17 or its checkpoints;
continuations are not independent restarts. Every unsuccessful trial remains.

See [all development results](../outputs/regular_pinn_recovery/all_development_report/REPORT.md)
and [locked assessment results](../outputs/regular_pinn_recovery/locked_assessment_report/REPORT.md).
Each report has plots with brief interpretation and machine-readable parameter
tables. [Protocol](REGULAR_PINN_RECOVERY_PROTOCOL.md) records sequential choices.

## What was implemented

- A shared 5-hidden-layer, 160-unit tanh conditional pricing PINN, for one or two
  variance factors. It predicts a correction to analytic expected integrated
  variance; Black's formula reconstructs option prices. It has no exact Fourier
  evaluation in its forward function or inverse fit.
- The correct one-/two-factor forward-measure pricing PDE, differentiated twice
  in state coordinates; exact terminal payoff, positive implied variance, shape
  penalties, strict factor ordering and Feller-feasible parameter transforms.
- A branchless `-expm1(-kappa*tau)/(kappa*tau)` fixed the MLX compiled
  higher-derivative failure. Raw-price AD and deterministic-variance tests check
  the transformed PDE independently.
- Standard synthetic price/IV-function training versus added Sobolev supervision
  of sensitivities to all 5/10 unit parameter coordinates. AdamW, weight decay
  1e-6, clipped gradient norm 5, fixed learning-rate schedules and saved histories.
- Identical trained weights copied into float64 PyTorch for inverse calibration,
  using bounded multistart SciPy least squares and analytic neural Jacobians.
  The old encoder plus exact-price polish remains a different, hybrid method.

The network is parameter-conditioned during forward training. At inverse use,
unknown parameters are optimized through that frozen network; generating truth
is not supplied to calibration. This is the existing regular Single Heston
workflow, not direct truth-supervised calibration at evaluation time.

## Data, coordinates and constraints

Core reference files are under `outputs/regular_pinn_recovery/{single,double}_data`:
131,072 candidate training quotes and 16,384 validation quotes per family.
Single training has 131,065 usable labels; all rejected candidates remain with
explicit masks. Double training and both validation sets are fully usable.
References use 128-node Fourier prices checked against 96 nodes; implicit Black
inversion gives IV and its parameter sensitivities in float64.

An 18,000-point PDE collocation pool is refreshed every 1,000 updates; 256 points
are used per optimizer step. The pool count is not the cumulative sampling count.
Ordinary anchor minibatches contain 1,024 quotes. Tau is calendar days / 365,
from 7/365 to 2 years. Half the sampled maturities are continuous, half use
30/60/90/180/365/730-day slices. These are synthetic scenarios, not exchange
expiry-day adjustments or a claim that all such stock maturities trade on NSE.

Coordinates are `x=log(F/K), v1[,v2], tau`, with known carry normalized into
the forward price. Canonical stored parameters are `[kappa,theta,sigma,rho,v0]`
per factor. **Factor 1 is slow, factor 2 fast** in this iteration; this is the
reverse naming order of the user's initial fast-first formulation, consistently
canonicalized throughout sampling, pricing, calibration and reporting.
Full numerical bounds are in `regular_pinn_data.DOMAIN` and the protocol.
These finite, Feller-feasible bounds are assumptions, not the entire Heston class.

An additional 1,024 independently sampled training surfaces (129,024 quotes,
seed 907721) supported the grouped-price control and parameter-aware loss trial.
All reference surfaces were numerically usable. A truncated Jacobian inverse
retained rank 10 in 961 surfaces and rank 9 in 63; omitted weak directions were
disclosed, not declared recovered. No validation/test parameters generated these
training surfaces. The parameter-aware loss did not improve actual recovery.

## Results — do not mix these evidence levels

**Development:** four separate fixed recovery surfaces selected checkpoints.
Double Sobolev scaled recovery RMSE improved from 0.163274 (standard) to 0.124257,
about 24%, but both passed 0/4 all-parameter cases. Extra low-LR training, lower
PDE weight and parameter-aware training did not beat that selected checkpoint.
The score is not an ordinary average percentage error: rho errors are scaled
by 0.5, other errors by their true positive values.

**Frozen assessment:** seed 907931, 12 new truths per family, monthly/rich
maturities, clean/1%-time-value-noise conditions, every third strike withheld.
Four frozen models, 192 model-condition fits and 960 starting-point attempts;
these reuse 24 distinct truths, not 192 independent parameter vectors.

| Selected Sobolev model | Clean six-expiry parameter target | All price + parameter gates |
|---|---:|---:|
| Single Heston, own-model truths | 10/12 | 1/12 |
| Double Heston, own-model truths | 0/12 | 0/12 |

Both Double Heston variants passed 0/12 all-parameter cases in each of all four
assessment conditions. In clean three-expiry DH cases, the Sobolev neural price
gate passed 12/12 but the parameter gate passed 0/12. This is precisely why a
close neural price fit cannot establish structural recovery. The strict price
gate is heldout RMSE <=1e-5 of spot; the parameter gate is <=5% for every positive
parameter and <=0.05 absolute for each rho. Keep exact repricing, neural pricing,
optimizer termination and parameter gates separate.

Assessment truths use central unit coordinates [0.1,0.9], approximately 10.7% of
the ten-dimensional unit cube by volume. Boundary robustness, repeated training
seeds, market calibration and a full common-data BS/SH/DH PINN comparison remain
unestablished. The BS rows are an analytic one-volatility price baseline only.

## Validity evidence and diagnosis

- 43 scoped regression tests passed (19 for the new regular pipeline); existing
  TorchScript deprecation warnings on Python 3.14 remain. This is not a claim
  that the complete historical repository suite was rerun.
- All 15 frozen-artifact audit checks passed, including row uniqueness, expected
  counts, recomputed gates, physical constraints, minimum calibration-SSE start
  selection, source/checkpoint hashes and zero exact train/validation/test
  parameter overlaps. 191/192 selected optimizers reported convergence.
- Calibrator tests corrupt heldout values without changing fitted results and
  forbid Fourier calls during fitting. Truth enters only generation and scoring.
- Train/validation IV errors are close: Double Sobolev 0.000336/0.000348. No
  strong memorization signal is observed on this distribution; this is not proof
  against all overfitting or distribution shift.
- Development exact Jacobian condition numbers in parameter-tolerance coordinates
  range about 1,469–15,806. Small surrogate errors project onto weak parameter
  directions and cause large inverse errors. Teacher sensitivity quadrature
  discrepancies were much smaller than the learned-function error.
- Fresh PDE diagnostics are finite, but the full wide-domain residual is not
  near zero. For selected Double Sobolev: near-smile RMSE 0.00660; wide-domain
  RMSE 0.17536. Terminal and identical-weight float32/float64 parity checks pass.
  These diagnostics do not certify a globally solved PDE.

Detailed files: `generalization_stage1.json`, `validation_diagnostics/`,
`physics_stage1_seed907611.json`, `physics_continuation_seed907611.json`,
`physics_grouped_seed907611.json`, and `locked_assessment_report/audit.json`.
The original stage-1 trainer source was recovered by exact SHA256 matching and
is embedded in `generalization_stage1.json`; newer runs have source snapshots.

## Code and safe next work

Core: `src/mentor_dh_pinn/regular_pinn.py`, `regular_pinn_data.py`,
`regular_pinn_torch.py`. Scripts are in `scripts/mentor_dh_pinn/`:
`prepare_regular_pinn.py`, `train_regular_pinn.py`, `assess_regular_pinn.py`,
`prepare_regular_pinn_surfaces.py`, the three diagnostics and the two report
generators. All commands expose `--help`; output directories must be new.

Training requires MLX on Apple Silicon; inverse inference uses identical-weight
PyTorch. Local Python: `/Users/dhruvaambhaikar/Documents/Options pricing/.venv/bin/python`.
Training and assessment entry points must run from the repository root.

Not yet attempted in this regular experiment: network-weight L-BFGS fine-tuning
or a float64 training phase. The inverse SciPy optimizer is not network-weight
L-BFGS. Those are sensible controlled next experiments before another major
architecture change. Extra dropout does not create missing price information
and introduces noise into PDE derivatives. If a factor-structured architecture
is explored, label its difference from this regular pricing-PINN honestly.

Do not tune to assessment seed 907931 and call it unseen again. Choose from the
declared development data, freeze new choices, use fresh assessment truths,
repeat training seeds and include boundary/noise cases. Retain failures and
all start logs. Do not substitute exact Fourier-polished parameters and call
them PINN recovery, nor weaken the already declared gates after seeing results.
No current recovery claim supports calling the project goal complete.

## Next iteration and repository publication

The user has now explicitly requested working toward all ten parameter targets
passing on the previously assessed 12 clean six-expiry cases. Those exposed cases
are development cases from now on; preserve the original assessment and require
fresh cases before another unseen-generalization claim. Do not feed their truth
parameters into initialization, training labels, or the calibration objective.

Float64 PyTorch PDE support and six independent tests have been added as
preparation for deterministic fine-tuning. The expanded scoped suite passes
49 tests. No L-BFGS network training result exists yet. Frozen assessment source
hashes refer to the source version at that assessment; newer PDE-support methods
are a later change, not a rewritten historical result.

The GitHub snapshot includes code, reports, regular PINN checkpoints, selected
hybrid-repair checkpoints, and small repair evaluation inputs. Regenerable
training arrays and local caches remain ignored. To regenerate the regular
reference files on a new checkout, use the existing preparation script with
`--factors 1 --seed 906110` for Single and `--factors 2 --seed 906210` for Double,
the default training/validation sizes, and the corresponding new output directory.
Generate extra grouped surfaces with seed 907721 only if reproducing those trials.
