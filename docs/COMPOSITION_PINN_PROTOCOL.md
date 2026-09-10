# Compositional price-PDE PINN — exploratory protocol

## Reason and scientific scope

The 17-layer and five-layer controls both recovered zero complete ten-parameter
vectors in each of two 12-case cohorts. Extra depth did not solve recovery.
This separately labelled variant composes two independent Single Heston price
PINNs. It does not replace the original architectures or change their gates.

For a normalized call c(x)=E[(exp(x+Y)-1)+], the log-return density satisfies
f(y)=c_xx(-y)-c_x(-y). Independent factor returns therefore give
c_double(x)=integral c_1(x+y) f_2(y) dy. Differentiable Gauss-Hermite quadrature
with 96 nodes implements this identity. There is no exact Heston pricer or ODE
solver in neural inference or inverse optimization. No density clipping or mass
renormalization is allowed. The component is trained on the Single Heston price
PDE; this is NOT direct training of the full Double Heston three-state PDE.

## Fixed training plan (declared before component training)

- Warm start: `single_sobolev_s17`; original five hidden tanh layers, width 160.
- Independent synthetic data seed 910021: 131072 training candidates, 16384
  validation candidates (seed 910022), and 18000 collocation points.
- Component physical box: kappa .2–12; theta and initial variance .006–.30;
  rho -.8–.5; sigma/sqrt(2*kappa*theta) .15–.90; maturity 7/365–2 years.
  This broader TRAINING box covers both factors. Inverse parameter maps remain
  unchanged, including the canonical slow-first/fast-second ordering.
- Record all rejected reference labels and retain original archives. Training
  uses only the archived numerical-reference usable mask. Never filter neural
  errors or prediction failures to improve reported metrics.
- Float64 AdamW, 5000 updates, seed 910031; learning rate cosine 1e-5 to 2e-7;
  decay 1e-6; anchor batch 512; PDE batch 128; existing Sobolev and PDE weights.
- Select lowest component validation IV RMSE among step 0 and every 500 steps.
  Recovery truths, calibration quotes and held-out quotes are not training inputs.

## Fixed pilot comparison

After freezing weights, run the 12 already-exposed clean six-expiry cases from
truth seed 927931. These are DEVELOPMENT evidence, not untouched generalization.
All models receive identical calibration quotes and identical held-out quotes.
Models: legacy Single PINN; broader component Single PINN; compositional Double
PINN. Use five blind LHS starts, 400 evaluations per start and the existing IV
objective. Retain all failures. Starts use the same declared case seeds but their
dimensions necessarily differ. Compare held-out neural prices AND independent
exact repricing of fitted parameters. Exact repricing is assessment-only.

DH recovery retains eight relative-error gates <=5% and two absolute correlation
gates <=.05; complete recovery means all ten pass. A Single Heston fit to DH data
has no corresponding true five-parameter vector; do not manufacture one. Pricing
superiority on DH-generated surfaces alone does not establish general superiority
over Single Heston. If the pilot fails, report it rather than asserting success.

## Required checks

Analytic Black composition and gradients; implied-volatility implicit gradients;
exact-reference composition identity; checkpoint roundtrip; finite-difference
neural parameter Jacobian; guarded exact-pricer exclusion and held-out corruption
replay. Audit density mass, martingale moment, negative densities and 96-vs-128
node changes without correcting them silently. Keep numerical failures visible.

The independent multifactor framework is described by
[Christoffersen, Heston and Jacobs (2009)](https://pure.au.dk/ws/files/17142435/rp09_34.pdf).
That paper's market results are not evidence for this neural implementation.

## Training outcome and evaluation freeze

The 5000-update run `component_adam910031` completed. The independent component
validation RMSE was .0003641528637 at step 0 versus .0004040459516 at step 5000;
no saved update beat step 0. The selected weights are unchanged from the legacy
Single checkpoint, so this continuation did NOT establish broader-domain
training improvement. The two Single controls therefore use identical weights;
the actual change being tested is the Double composition.

Frozen composition checkpoint SHA256:
`70e691241fcf32c8824d2ca5e81a8ad1e663fb04f77a94103966467107687d53`.
Same-quote comparison: `composition_comparison910031`, two CPU workers, all
twelve exposed cases. The component selection was fixed before those fits.
Six reference-pricer entry points are blocked during every inverse fit. Case 0
is fully replayed for each model, all five starts, with held-out inputs replaced
by NaN. Every case retains all starts, failed predictions and density diagnostics.

The exact-reference composition check initially failed at 128 GH nodes: the
fixed 128-node Fourier reference aliases in the farther tails introduced by the
larger GH rule (density mass/martingale errors expose this). Converged 32/48-node
reference composition agrees with direct Double Heston at the original 1e-8
absolute test tolerance. This does not change production neural nodes (96) or
its separate 128-node quadrature audit. The failure is retained here, not
interpreted as evidence that more reference quadrature is always better.

## Test scope and broader checkout issues

All 52 tests in the regular/composition PINN scope passed. A whole-checkout run
first stopped at collection because the legacy Single Heston source directory
was not on the import path. With `PYTHONPATH=single_heston_pinn/src`, it completed
with **629 passed, 43 failed, 1 skipped**. Its XML is retained at
`outputs/regular_pinn_recovery/testing_composition_full_path910031/pytest.xml`.
Do not call the whole checkout fully validated.

Reported failures include missing historical NSE/evidence artifacts, preserved
hash mismatches, a runtime test pinned to Windows/Python 3.13 rather than this
Mac/Python 3.14 environment, an unavailable openpyxl dependency, old representation
schema mismatches, and a canonical price fixture differing by up to 3.27e-12
against its fixed 1e-12 tolerance. None of those fixtures, hashes or thresholds
has been rewritten to make the suite pass. Three training tests also encounter
a Float/Double mismatch in the combined run; the existing
`tests/test_unified_calibrator.py` sets the global Torch default to float64 at
import time. The new composition code does not change the global dtype.
Those three training tests pass when rerun in isolation (XML:
`testing_composition_legacy_isolated910031/pytest.xml`). This supports the
cross-test global-state diagnosis; no unrelated production model was changed.

## Separately predeclared component loss-balance trial

While the fixed 12-case composition pilot continues, a second independent-weight
trial changes ONLY training settings, not that comparison or its checkpoints.
`component_balanced910071` starts again from the selected step-0 component,
uses the same archived data and 18000-point PDE pool, seed 910071, 10000 AdamW
updates, initial learning rate 5e-5 cosine-decaying to 2e-7, and PDE weight .02
instead of .2. Shape weight remains .05; anchor and sensitivity terms unchanged.
Select component validation IV RMSE only at step 0 and every 1000 updates.
No recovery-case data enters this trial. This is a development loss-balance
experiment, not an independently assessed improvement. Any subsequent recovery
comparison must be separately labelled and retain the first pilot's failures.

Before its training results are inspected, the follow-up rule is fixed: only if
the selected validation IV RMSE improves by at least 1% over .0003641528637,
package the new component and run the same full 12-case, five-start comparison,
using the same 400-evaluation cap. Include both legacy Single and newly selected
Single controls. Otherwise retain the negative training result without spending
another full calibration budget on numerically identical selected weights.
The unchanged gate definitions and exposed-development label apply in either case.

## September 10 continuation

The first comparison completed with 57/120 individual gates, 0/12 complete
recoveries, 12/12 neural price gates, and 2/12 exact-repricing price gates.
Both neural and independently repriced held-out RMSE beat the identical legacy
Single controls in all 12 DH-generated cases. This is conditional pricing
evidence, not universal model superiority or successful parameter recovery.
Report: `outputs/regular_pinn_recovery/composition_report910031/REPORT.md`.

The loss-balance trial completed 10000 updates, selecting step 10000 at component
validation IV RMSE .00021003359079454478 (about 42% below its initial error).
The predeclared follow-up criterion is met. Frozen composition SHA256:
`312fbe50e52d70cee09c7e264e23e21df33f50bc32eee1886470de08addc09eb`.
The second full comparison is `composition_balanced_comparison910071`.

Before the second assessment, terminal payoff handling was added explicitly.
Density diagnostics and IV require positive maturity; price alone accepts zero
maturity and returns the exact payoff. Against the first assessment's source
snapshot, every IV and every input gradient was bitwise unchanged on all 12
fitted positive-maturity surfaces. First-assessment sources remain archived.

A separately frozen raw-price PDE diagnostic uses 512 new states, seed 910101,
eight states per derivative batch. It checks the full Double Heston PDE directly
through the composition, despite training only the component PDE. Its results
are diagnostic, not checkpoint-selection inputs or evidence of direct full-PDE
training. A zero-vol-of-vol analytic-limit test checks the residual implementation.

That audit completed: raw full-PDE RMSE fell from .0001436217 to .0000961513
for the balanced weights. All values were finite. At the fixed 1e-12 shape
threshold there were no convexity or upper-bound violations. The old/new models
had 14/13 negative-calendar states and13/13 lower-bound violations respectively,
all in the wide-moneyness stress subset. Maximum lower-bound shortfalls were
5.84e-9 and4.05e-9 normalized-price units. These small numerical violations remain
flagged; no clipping, renormalization or threshold relaxation was applied.
All54 focused tests including terminal payoff and raw-PDE checks pass (XML:
`outputs/regular_pinn_recovery/testing_composition_final910101/pytest.xml`).

## Reproduction entry points

Run from the repository root with its Python environment. Output paths must be
new; the scripts refuse to overwrite completed/partial experiment evidence.

```sh
python scripts/mentor_dh_pinn/compare_composition_pinn.py \
  --checkpoint outputs/regular_pinn_recovery/single_sobolev_s17 \
  --checkpoint outputs/regular_pinn_recovery/component_balanced910071 \
  --checkpoint outputs/regular_pinn_recovery/composition_balanced910071 \
  --out outputs/regular_pinn_recovery/NEW_COMPARISON --workers 2
python scripts/mentor_dh_pinn/report_composition_pinn.py \
  --assessment outputs/regular_pinn_recovery/NEW_COMPARISON \
  --training outputs/regular_pinn_recovery/component_balanced910071 \
  --physics outputs/regular_pinn_recovery/composition_physics910101.json \
  --out outputs/regular_pinn_recovery/NEW_REPORT
```

`case_*.json` contains all starts, selected physical parameters, truth (assessment
only), the same quote split, neural and exact reprices, and density diagnostics.
`PARAMETERS.md` and `audit.json` give the full team handoff; `REPORT.md` includes
plots with interpretation. These results do not update NSE/PostgreSQL/Kaggle.

## Training-only conditioning diagnostic

After freezing both variants, the first 32 surfaces of the preserved independent
training archive `double_surface_train907721/surfaces.npz` were scored without
filtering/resampling. Average IV RMSE improved from .0001383094 to .0000948106;
the RMS of the local tolerance-coordinate bias proxy improved from9.8787 to7.8589,
but its median largest component worsened from5.3966 to5.9595. This proxy is
the archived reference-Jacobian pseudoinverse times neural IV error, NOT actual
recovered parameters. Truncated singular directions are not measured. No exact
pricer, calibration, or checkpoint selection ran in this diagnostic. Evidence:
`outputs/regular_pinn_recovery/composition_bias910101.json`.

If further development proceeds, reuse the existing independent grouped-surface
data and sensitivity-loss formulation rather than simply adding more layers.
Adapting that training-only loss to the composed price PINN remains unimplemented
and is not a promised solution. IV is derived from price, not an independent
measurement; ordering factors removes label swapping, not all practical ambiguity.

## Final comparison outcome — September 10

| Trial | Individual parameter gates | Complete recovery | Neural price gate | Exact repricing gate |
|---|---:|---:|---:|---:|
| First composition | 57/120 | 0/12 | 12/12 | 2/12 |
| Loss-balanced composition | 50/120 | 0/12 | 12/12 | 0/12 |

Both compositions beat both corresponding Single controls on each of the12
held-out neural-price and exact-repricing comparisons. This is exposed,
DH-generated development evidence, not a universal performance claim. The
loss-balanced model's median exact-repricing RMSE/legacy-Single ratio is.06979
(first trial .09077), but individual recovery gates worsen and exact-repricing
threshold passes fall to zero. Neither variant meets the ten-parameter goal.
The default architecture is unchanged and no new model is promoted as solved.

All36 selected fits in the second comparison satisfy the optimizer's convergence
criterion. All180 starts finish without recorded failures or budget exhaustion.
That does not prove a global optimum. The all-start held-out corruption replay
is identical for each model on case0, with all reference entry points guarded.
Checkpoint/source hashes are rechecked and every metric, parameter gate and
minimum-SSE start selection is recomputed in the report audit.

Observed median per-case fit times were about.24seconds for Single versus
158seconds for first composition; about.25seconds versus173seconds in the second
trial. These CPU runs were concurrent with other diagnostics, not a controlled
speed benchmark. The new composition is therefore NOT a demonstrated runtime
improvement; the observed advantage is conditional pricing accuracy only.

Latest report: `outputs/regular_pinn_recovery/composition_balanced_report910101/REPORT.md`.
Its `PARAMETERS.md` includes true/estimated Double parameters and both Single
models' fitted parameters (with no invented Single truth). Its `audit.json`
includes physics and conditioning diagnostics. All55 scoped tests now pass,
including stochastic Fourier-reference and deterministic-limit PDE checks:
`outputs/regular_pinn_recovery/testing_composition_stochastic910101/pytest.xml`.
Broader checkout failures remain recorded above. All jobs have finished; all
new work remains local and unpushed. NSE, PostgreSQL and Kaggle are untouched.
