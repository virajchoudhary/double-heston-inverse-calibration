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
