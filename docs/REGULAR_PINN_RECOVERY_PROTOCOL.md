# Regular PINN parameter recovery experiment

This experiment follows the user's clarified comparison: Black–Scholes, Single
Heston and Double Heston calibrated through regular pricing PINNs. The existing
Single Heston method first trains a conditional price function and then optimises
parameters through that frozen network. The new implementation uses that same
workflow for one and two CIR factors. The old Set-Transformer / exact-price polish
results remain a different method and cannot establish this method's accuracy.

## Fixed design before new training or assessment

- Shared smooth five-layer, 160-neuron network, implied-total-variance output and
  analytic Black price reconstruction; correct one-/two-factor Feynman–Kac PDE.
- Independent synthetic references: 131,072 candidate training and 16,384 validation
  points per model, 128-node Fourier evaluation checked against 96 nodes (1e-9).
  Every rejected candidate remains in the archive with an explicit usability mask.
- Parameter draws and quote/state coordinates come from seeded Latin hypercubes.
  Full parameter ranges are recorded in `regular_pinn_data.DOMAIN` and manifests.
  These are declared finite training ranges; results do not cover every legal
  Double Heston parameter vector. Factor ordering, positivity, strict Feller and
  the inherited joint correlation constraint hold throughout this experiment.
- PDE collocation pool: 18,000 points, sampled in minibatches; refreshed every
  1,000 steps. The cumulative number of sampled points is therefore larger than
  the simultaneous pool. Independent PDE validation will use another seed.
- Standard arm: supervised reference correction plus PDE and static constraints.
  Enhanced arm adds derivatives of the correction with respect to all 5/10 unit
  parameter coordinates (Sobolev supervision), computed from training references
  through implicit Black inversion. Those derivatives are training labels, never
  inputs supplied when calibrating an unknown assessment surface.
- Both arms use AdamW, weight decay 1e-6, gradient norm limit 5, 250-step warmup
  and cosine schedule. Initial development budget: 12,000 steps per arm/model,
  1,024 anchor and 256 physics points per step, learning rate 1e-3. Sensitivity
  coefficient 0 for standard, 0.2 for enhanced; PDE coefficient 0.2.
- The first development seed is 17. Architecture/weight changes may be made from
  development validation. Replicate the selected configuration with seeds 29/43
  before a final claim of reliability. Retain all configurations and outcomes.
- Select checkpoints using physical-parameter recovery on four fixed, separate
  synthetic validation surfaces, not price-error proxies. Positive-parameter
  errors are divided by their true values; correlations by 0.5. These validation
  labels are used only for scoring/checkpoint selection. They never enter fitting.
- Calibration minimises errors in neural implied volatilities using analytic
  neural Jacobians and bounded multistart least squares. It may not evaluate
  Fourier prices, refine against an exact engine, or see true parameters.

## Assessment and interpretation

Final assessment will use fresh seeds declared in the final assessment manifest
after training choices are frozen, at least 12 new parameter sets per model and
training seed. Identical surfaces are shared by compared training seeds. Assess
three monthly maturities (30/60/90 days) separately from a six-maturity synthetic
ladder (30/60/90/180/365/730 days). This ladder is not a claim about availability
of these expiries for a particular NSE stock. Every third strike is withheld.

Assess clean and 1% Gaussian noise relative to option time value, with intrinsic
value computed from known geometry. Do not truncate/resample noise or discard
invalid outcomes. Report calibration and held-out neural prices, exact-model
repricing at the same estimated parameters, and all individual parameter errors.
No exact price enters the deployed fit or its choice of starting point.

The clean recovery target is relative error <=5% for every positive physical
parameter and absolute error <=0.05 for every correlation. The held-out price
target is RMSE <=1e-5 of spot. Report all-case pass rates; a good price fit with a
failed parameter target remains a failed recovery. Report successful optimiser
termination separately, along with sensitivity/conditioning and numerical errors.
The noisy runs stress recovery; exact truth cannot be guaranteed from noisy data.

Development recovery surfaces and the initial assessment sampler use unit
coordinates in [0.1, 0.9], not the full training cube. In ten dimensions that
central region is only about 10.7% of the unit-cube volume. Any results from this
sampler must be called central-domain evidence; they cannot establish recovery
near the full parameter bounds. A separate full-domain/boundary stress assessment
is required before a domain-wide reliability claim.

Own-model synthetic generation tests recovery of each model's actual parameters.
Black–Scholes fitted to Double Heston prices is a misspecified pricing comparison:
its fitted volatility is not a recovered Double Heston parameter. Likewise,
Single Heston estimates on Double Heston surfaces have no generating Single
Heston truth labels. All synthetic data is explicitly labelled; no NSE records
are generated or changed by this study.

The current assessor covers own-model Single/Double Heston recovery and an
analytic Black–Scholes-on-Double-Heston price baseline. It is not the completed
three-PINN research comparison, and does not yet include shared-surface Single
Heston-on-Double-Heston pricing or a trained Black–Scholes PINN.

## Runtime correction, 2026-09-07

Use the exact branchless `-expm1(-kappa*tau)/(kappa*tau)` on the strictly positive
training domain. Removing the piecewise Taylor implementation resolves the MLX
compiled higher-derivative failure without changing the governing PDE. Independent
raw-price AD and zero-vol-of-vol checks validate the transformed PDE. Calibration
uses an identical-weight float64 PyTorch copy, not a different pricing model or
an exact-pricer refinement. A 200-step runtime pilot is not recovery evidence.

## Development continuation, fixed before running

The 12,000-step Double Heston sensitivity arm improves the fixed four-surface
scaled parameter RMSE from 0.16327 (standard) to 0.12426, but does not meet the
all-parameter recovery target. Single Heston sensitivity supervision improves
the corresponding score from 0.02548 to 0.01253. These are selected development
checkpoints, not unseen results or simple average percentage errors.

Next controlled experiment: continue the selected Double Heston sensitivity
checkpoint for 20,000 additional steps, constant learning rate 2e-5, sensitivity
weight 0.2. Compare PDE coefficient 0.2 versus 0.02 with all other settings and
initial weights fixed. These are continuations, not independent random seeds;
AdamW moments restart and the initial checkpoint hash is recorded. Preserve both
outcomes and check independent PDE residuals so improved price fitting is not
misrepresented as uniformly improved physics. Selection still uses the original
four validation surfaces; no final assessment is used to choose this change.

## Parameter-aware synthetic training experiment

The train/validation error ratios are approximately one, whereas Jacobian
diagnostics show strong amplification of small forward errors. To target this
approximation bias, generate 1,024 additional TRAINING parameter surfaces from
fresh LHS seed 907721, each with 126 fixed quotes (129,024 new training examples).
No validation truths enter this file. All 1,024 references pass price and
sensitivity numerical checks. The preconditioner is the pseudoinverse of the
exact IV Jacobian in physical recovery-tolerance coordinates. Singular directions
below absolute IV threshold 1e-6 (or relative 1e-10) are omitted: 961 surfaces
retain rank 10 and 63 retain rank 9. Omission is numerical regularization, not
proof that the tenth direction is recovered. All surfaces still receive price
loss; none is removed from future assessment denominators.

Compare two 12,000-step continuations of the same stage-1 Double Sobolev checkpoint:
both use constant LR 2e-5, PDE weight .2, sensitivity weight .2, and eight grouped
surfaces per minibatch in addition to the original training samples. Both add
grouped IV MSE weighted .1 after scaling IV errors by .01. The control has zero
parameter-aware weight; the experimental arm adds .001 times
`mean(log1p((B @ IV_error)^2))`. This is a robust, local linear proxy for
parameter distortion during TRAINING, not an inverse parameter result. Actual
validation recovery remains the unchanged, truth-blind neural-only calibration.
Do not infer success from decreasing this proxy, or credit the Fourier teacher
with doing the deployed calibration. These runs are not independent restarts.

## Locked assessment 1 (after the above runs completed)

The continuations and grouped-surface trials did not beat the original selected
Double Sobolev checkpoint on validation recovery. Retain them as negative results.
Freeze four existing selected checkpoints: Single standard (step 6000), Single
Sobolev (12000), Double standard (12000), Double Sobolev (12000), all seed 17.
Evaluate new assessment seed 907931, 12 truths per model family, both maturity
sets and both noise levels, five fixed LHS starts and 400 evaluations per start.
Keep every failure and the analytic BS pricing baseline. This is a previously
unseen central-domain assessment for these frozen models, not a full-domain or
multi-training-seed reliability claim. No model changes may be chosen using this
assessment and still described as independently tested on these same cases.

## Deterministic network-weight L-BFGS, 2026-09-08

The user requested all ten targets on the exposed 12 clean six-expiry cases.
These cases are now development evidence, not an unseen test. Neither their
truth vectors nor withheld quotes enter training or the inverse objective.
The old assessment remains unchanged. The repository snapshot was pushed to
`Double/Single-heston` at `3927ad7ff907504852423864afdd4d18ebf34298` before this phase.

Run `double_lbfgs908021` continued the selected Double Sobolev weights for
1,000 network-weight L-BFGS iterations, using 16,384 fixed training anchors and
all 18,000 original collocation points (seed 908021). Chunked sums evaluate the
entire fixed objective on every call. PDE relevance weights and derivative
normalizers are fixed numerically, not recomputed with stop-gradient. No
dropout, clipping, resampling, Adam learning rate or weight decay applies in
this phase. Full-objective directional checks preceded training. Effective
settings corrections explicitly preserve and supersede stale inherited-Adam
metadata in the original run config; original source snapshots are retained.

The objective fell from 0.00281701 to 0.00131633. Full validation IV RMSE fell
from 0.00034849 to 0.00027184, but original-four recovery RMSE worsened from
0.124257 to 0.201380. The unmodified starting weights remain selected. The
optimizer reached its declared iteration limit, not its convergence criterion.
This is a negative parameter-recovery result, not successful fine-tuning.

### Next controlled trial, declared before running

Keep the same regular pricing-PDE network and selected Double Sobolev starting
weights. Use the same fixed anchors, 18,000 collocation points, and all 1,024
independent training surfaces from seed 907721. Both arms add grouped IV MSE
with weight .1 after scaling IV errors by .01. The experimental arm adds the
quadratic local linear parameter-bias loss `mean((B @ IV_error)^2)`, replacing
the earlier logarithmically saturated proxy. Freeze its coefficient once,
using only initial training gradients, so its gradient norm is half the
base-plus-grouped-price gradient norm. Control coefficient is zero. Use
400 L-BFGS iterations per arm and original-four recovery checkpoint selection;
retain the initial checkpoint as a candidate and all unsuccessful outcomes.

`B` is the existing training-only tolerance-scaled Jacobian pseudoinverse.
Its rank is 10 on 961 surfaces and 9 on 63: the omitted direction is not
silently called supervised or recovered. Lower proxy loss is not an actual
parameter pass. No reference-pricer evaluations at trial parameters are
allowed in deployed inverse calibration. Evaluate selected weights on the
same explicitly exposed 12 cases only after training and selection finish;
fresh assessment cases remain necessary for any generalization claim.

### Paired result and derivative-consistency follow-up

Both 400-iteration arms completed. Grouped-price control ended at recovery RMSE
0.141308; quadratic bias ended at 0.220597. Neither beat the unchanged initial
score 0.124257, so both selected iteration zero. The quadratic coefficient was
2.5779730109365333e-6, fixed from initial training gradients, not tuned afterward.

A separately labelled neural self-consistency diagnostic on 12 new draws
(seed 908171) recovered all ten parameters in every case from the frozen
network's own IV outputs. Maximum heldout IV RMSE was 3.97e-15. This isolates
the optimizer's ability to invert that neural function; it is NOT recovery
against exact Double Heston targets and does not count toward the user's gate.

Next, retain the same initial weights, anchors, collocation and grouped training
surfaces; compare against the completed grouped-price control. Replace quadratic
bias with `mean(||B @ (J_neural_unit @ T) - P||^2)`, where
`T = solve(dp/du, diag(parameter_tolerances))` and P is the projector onto the
retained column space of B. All constants use TRAINING units only. Check stored
ranks, projector identities and scalar gradients. Freeze the new coefficient
to half the initial base-plus-grouped-price gradient norm, as before. Use
400 L-BFGS iterations, original-four checkpoint selection, and unchanged
heldout-strike calibration. No quadratic bias term is combined into this arm.
It targets derivative error in weak parameter combinations; success must still
be established by actual parameter recovery, not by the derivative proxy.

### Genuine float64 weight-training trial — declared before execution

The preceding MLX refinements did not improve the selected recovery checkpoint.
They used float32 network values and gradients despite float64 optimizer vectors
and subsequent inference. Test the existing, independently PDE-tested PyTorch
copy with float64 weights, gradients and AdamW state, without changing its
5-hidden-layer/160-unit regular price-PDE architecture. This is a precision and
training-protocol continuation, not an isolated causal test of dtype alone.

Warm start `double_lbfgs908021` (unchanged best original Sobolev weights). Reuse
all 131,072 clean `double_data/train.npz` labels, all 16,384 validation quotes,
and a fixed 18,000-point collocation pool. Seed 909111; 1,000 AdamW steps;
512 training anchors and 128 collocation points per update; learning rate
1e-5 to 2e-7 cosine decay; weight decay 1e-6; no gradient clipping or dropout.
Use the original correction MSE scaled by .05, Sobolev weight .2 with
training-only RMS normalization floored at .02, PDE normalized Huber weight
.2 and shape weight .05. Freeze the initial PDE relevance weights and full-pool
normalizer. Preserve full float64 checkpoints; never quantize back through MLX.

Select every 250 steps, including step zero, solely by full validation IV RMSE.
Neither original-four recovery truths nor exposed-twelve truths enter this
training/selection phase. Validation has been used previously and is not an
unseen test. Run a two-step operational pilot and objective-gradient checks
before the declared run; pilot weights do not become its warm start. After
selection, assess all twelve exposed clean rich cases using the unchanged
five blind starts, 400-evaluation budget and parameter/price gates. Preserve
every trial regardless of outcome. Float64 does not guarantee identifiability.

The 1,000-step float64 trial selected step 1,000: validation IV RMSE improved
from 0.0003484951214 to 0.0003410810733, but exposed-case recovery was **45/120
individual parameters, 0/12 complete and 0/12 joint cases**, below the original
48/120 baseline. Neural held-out price gates passed 8/12, exact repricing 0/12.
All 75 scoped tests passed. This does not demonstrate a precision-only solution.

One extended float64 continuation is declared next: warm start that selected
step, train 5,000 additional AdamW steps with a fresh optimizer and the same
seed 909111, learning-rate endpoints, losses, minibatches and fixed pool;
select by full validation IV every 500 steps, including the starting weights.
Assess exposed recovery only after selection is complete. This is additional
training on the same corpus, not new independent examples or a new restart.
The fresh RNG repeats early sampled minibatches; this is disclosed, not a claim
of an independent draw. No recovery-based stopping/selection or gate changes.

Precision metadata clarification: PyTorch AdamW's parameter-sized first/second
moments are float64 here; its scalar step counter uses the backend default
(float32). The original trial's broad phrase "optimizer state" should be read
with that clarification. It does not affect parameter/gradient precision;
original config/checkpoint hashes remain untouched.

The extended run completed all 5,000 additional steps in 290.2 seconds and
selected step 5,000 at validation IV RMSE 0.0003354423479. All 131,072 training
quotes were visited. Recovery remained **45/120 individual, 0/12 complete and
joint cases**; neural price gates 8/12, exact repricing 0/12. This is a negative
parameter-recovery result, not a successful refinement. Both stages passed
all-twelve-case held-out corruption replay with reference-pricer entry points
blocked, exact tensor/selection/hash audits and the 75-test scoped suite.

Post-fit local diagnostics found lower neural calibration SSE at each selected
estimate than at its generating truth in all twelve cases, in both stages.
These diagnostics did not feed back into training, selection or fitting.
They support approximation-bias concerns; they do not demonstrate global
non-identifiability or provide new fitted parameters. Full results and limitations
are in `float64_combined_report909111` and the two `float64_*_bias909111` artifacts.
