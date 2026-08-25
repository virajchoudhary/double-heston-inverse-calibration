# Model 3 genuine PDE-informed Double Heston protocol

Status: `MODEL3_PDE_PILOT_READY_AFTER_AUDIT_FIXES`. The scientific architecture,
canonical PDE, data boundary, loss, experiment stages, metrics, and
interpretation rules remain frozen before any Model 3 training result exists.
Commit `f34a4d3` had been marked pilot-ready, but an adversarial pre-pilot audit
found blocking execution defects; readiness was revoked before any Stage-A or
scientific execution. The execution-layer corrections are a pre-result repair.
A fresh adversarial review marked all six execution blockers and both follow-up
findings resolved. This status authorizes no pilot execution.

## 1. Motivation and claim boundary

Model 1 is the ordinary ANN. Model 2 is the constraint-informed,
differentiable-repricing-informed inverse model; its Fourier-repricing loss is
not a PDE residual and its historical class name must not be used to redefine
it. Model 3 asks a new question: does an explicit canonical Double Heston
pricing-PDE residual improve recovery, robustness, structural behavior, or
stability beyond those frozen baselines?

A positive, mixed, null, or excessive-cost result is valid. Low repricing error
never proves unique parameter recovery. Practical non-identifiability remains a
retained finding.

## 2. Canonical dynamics

The production engine in `src/double_heston.py` is the sole dynamics source of
truth. Its characteristic function contains one deterministic log-spot drift
and two additive variance-factor exponents. With risk-free rate `r`, carry or
dividend yield `q`, spot `S`, slow/fast variances `v_s` and `v_f`, the
risk-neutral system is:

```text
dS/S = (r-q) dt
       + sqrt(v_s) dW_0s
       + sqrt(v_f) dW_0f

dv_s = kappa_s (theta_s-v_s) dt + sigma_s sqrt(v_s) dW_s
dv_f = kappa_f (theta_f-v_f) dt + sigma_f sqrt(v_f) dW_f

E[dW_0s dW_s] = rho_s dt
E[dW_0f dW_f] = rho_f dt
E[dW_s  dW_f] = 0
```

Canonical order is:

```text
kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow,
kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast
```

Structural constraints are strict positivity of kappa, theta, sigma, and v0 in
both factors; `kappa_slow < kappa_fast`; strict positive Feller gap
`2*kappa*theta-sigma^2` in each factor; each correlation inside `(-1,1)`; and
`rho_slow^2+rho_fast^2 < 1`. Provisional hard numerical safety bounds remain
separate from structural validity and are never silently clamped.

## 3. Exact pricing PDE

Let `V(t,S,v_s,v_f)` be the European option value. In calendar time `t`, with
terminal time `T`:

```text
V_t
+ (r-q) S V_S
+ kappa_s(theta_s-v_s) V_vs
+ kappa_f(theta_f-v_f) V_vf
+ 0.5(v_s+v_f)S^2 V_SS
+ rho_s sigma_s v_s S V_Svs
+ rho_f sigma_f v_f S V_Svf
+ 0.5 sigma_s^2 v_s V_vsvs
+ 0.5 sigma_f^2 v_f V_vfvf
- rV
= 0.
```

There is no `V_vsvf` term because the implemented variance factors are
independent. The payoff condition is
`V(T,S,v_s,v_f)=max(S-K,0)` for a call and `max(K-S,0)` for a put.

Using time-to-maturity `tau=T-t` gives `V_tau=-V_t`. The implemented residual
is therefore:

```text
R =
V_tau
- [
    (r-q)S V_S
    + kappa_s(theta_s-v_s)V_vs
    + kappa_f(theta_f-v_f)V_vf
    + 0.5(v_s+v_f)S^2V_SS
    + rho_s sigma_s v_s S V_Svs
    + rho_f sigma_f v_f S V_Svf
    + 0.5 sigma_s^2 v_s V_vsvs
    + 0.5 sigma_f^2 v_f V_vfvf
    - rV
  ].
```

At `v_i=0`, diffusion and mixed-derivative coefficients degenerate to zero but
CIR mean-reversion drift remains. No artificial Dirichlet condition is invented
there. Finite sampling stays strictly inside the variance domain. European
discounted no-arbitrage bounds constrain the forward output, and expiry payoff
is enforced exactly by construction.

## 4. Architecture

The selected narrowest genuine design is a conditional forward pricing PINN
coupled to an inverse R2 encoder and trained jointly.

| Component | Contract |
|---|---|
| Inverse input | frozen 100-feature R2 vector |
| Inverse output | ten canonical parameters through `DoubleHestonConstraintMap` |
| Forward state | `log(S)`, `v_s`, `v_f` |
| Forward contract | `log(K)`, `tau` years, `r`, `q`, call/put indicator |
| Parameter conditioning | eight structural parameters only |
| Excluded conditioning | predicted `v0_slow` and `v0_fast` |
| Forward output | dollar price in discounted European no-arbitrage bounds |

At observed quotes, the inverse-predicted initial variances define the PDE
state pair `(v_s,v_f)=(v0_slow,v0_fast)`. They are not also given as forward
conditioning. Otherwise the network could ignore the state variables and reduce
the physics term to repricing. At other collocation points, the state variances
are sampled independently subject to the conditioned domain.

Graph caveat: those observed-state values are detached before becoming PDE
coordinate leaves because the reverse-mode operator differentiates with respect
to independent state coordinates. Structural parameter heads therefore receive
PDE/reconstruction coupling through conditioning and coefficients, while `v0`
heads are directly trained by supervised parameter loss under this architecture.
This is not a claim that PDE physics directly regularizes the `v0` heads; any
architecture change requires a separately reviewed pre-research decision.

The skeleton constructs float64 smooth MLPs with widths `128,128,64`, tanh
activation, and a hard bounded output map. The shared R2 builder's float32
features are explicitly upcast at the Model 3 inverse boundary. These choices
support second derivatives and are frozen before results.

## 5. Data and representation

The primary dataset is immutable:

```text
data/final_r2_clean_10000/surfaces.jsonl
SHA-256 148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6
```

Splits come only from stored per-surface metadata: 7,500 train, 1,250
validation, and 1,250 untouched test surfaces. The shared R2 feature builder is
unchanged: masked normalized prices, mask, maturities, rates, and carries in
canonical slot order. Masked quotes are never imputed. Target normalization is
fit on train data only. Real-market observations never update Model 3 weights.

Issue #34 artifacts stay on `codex/r2-noise-recovery`. Their numeric outcomes
are motivation, never training, validation, hyperparameter, seed, stopping, or
selection input for Model 3.

## 6. Frozen loss

For weights fixed before any result:

```text
L =
  1.00 * L_parameter
+ 1.00 * L_reconstruction
+ 0.10 * L_pde
+ 0.00 * L_terminal_diagnostic
+ 0.00 * L_boundary_diagnostic
```

- `L_parameter`: MSE in standardized target units, using train-only statistics.
- `L_reconstruction`: masked MSE between predicted dollar price divided by
  surface spot and observed normalized price.
- `L_pde`: mean squared `R/max(abs(V),1)` over float64 autograd collocations.
- `L_terminal_diagnostic`: payoff mismatch at `tau=0`; weight is zero because
  the output map enforces expiry payoff exactly.
- `L_boundary_diagnostic`: violation of discounted no-arbitrage bounds; weight
  is zero because the map already enforces those bounds.

Structural constraints are enforced by construction through the existing map;
there is no extra penalty. Validation may drive early stopping only under the
frozen rule. It may not tune architecture, weights, seeds, optimization, or
collocation counts.

## 7. Collocation and terminal design

All physics tensors are float64 and every state variable must be an autograd
leaf. The operator rejects non-leaf or detached states and refuses missing graph
dependencies rather than replacing a derivative with zero.

For each observed spot `S0` and conditioned long-run variance `theta_i`:

| Variable | Domain |
|---|---|
| spot | uniform `[0.50*S0,1.50*S0]` |
| slow variance | `[0.05*theta_s,min(2*theta_s,slow hard ceiling)]` |
| fast variance | `[0.05*theta_f,min(2*theta_f,fast hard ceiling)]` |
| maturity | uniform `[7/365,180/365]` years |

For every interior or terminal point, its source index identifies only a
surface. A separate seeded draw selects one slot from that surface's full set of
observed/unmasked canonical R2 slots; strike and call/put are read as
`contract[surface_index, canonical_slot_index]`, never as a flattened-contract
index. Rates and carries remain surface-level as represented by the frozen R2
loader. Observed slots are used so physics diagnostics stay aligned with the
eligible observation geometry; masked quotes remain excluded and are never
imputed. This v1.1 clarification was made before any Stage-A result existed and
does not change populations, epochs, optimizer settings, point counts, seeds,
loss weights, or the canonical PDE. Pilot batches use 16 interior and 8 terminal
points per surface; research runs use 32 interior and 8 terminal points. CPU
generator seed is 3407.

Below seven days, a C2 polynomial blends the bounded base to exact expiry
payoff. At or above seven days, the shortest supported tenor, terminal weight is
exactly zero, so the full bounded-base interval remains representable.

## 8. Split and leakage rules

Training reads stored train rows only. Validation selects checkpoints by minimum
total validation loss with patience 15 evaluations in the frozen run. The test
split is opened once after all seeds and checkpoints are final.

Forbidden before that point:

- any test forward pass or metric;
- use of test information for architecture, normalization, loss, weights, seed,
  stopping, checkpointing, or reporting decisions;
- Issue #34 positive-noise numeric outcomes as selection signals;
- real-market weight updates;
- reopening R2-vs-R3;
- changing the final 10k dataset or primary evidence.

Any later development-noise set must be derived only from train/validation data
with fresh deterministic keys unrelated to Issue #34.

## 9. Metrics and baselines

After freeze, compare Model 3 with ordinary ANN, Model 2, and traditional
calibration where the population supports a fair comparison. Report these
families separately:

1. parameter recovery, including range-scaled RMSE and per-parameter errors;
2. clean-latent repricing using the unchanged production pricer;
3. observed/noisy repricing when a later noise evaluation is separately frozen;
4. structural validity and constraint diagnostics;
5. cross-seed stability and prediction dispersion;
6. training/inference runtime and cost;
7. degradation curves for any separately authorized robustness extension.

Repricing quality is never substituted for parameter recovery.

## 10. Result interpretation

Exactly one of these conclusions may be supported by the frozen evaluation:

1. Model 3 improves parameter recovery;
2. Model 3 improves robustness without clean-accuracy improvement;
3. Model 3 improves structural validity or stability;
4. Model 3 provides no measurable benefit;
5. Model 3 is too expensive to justify.

Thresholds and architecture cannot change after seeing outcomes. Failure also
includes non-finite gradients, irreproducible identities, unstable physics
diagnostics, or inability to complete within budget.

## 11. Two-stage experiment

### Stage A development pilot

Purpose: implementation and numerical-trainability validation only. This is not
a research result. Use the first 240 train and first 40 validation surfaces in
frozen file order, development seed 4207, batch size 16, AdamW learning rate
`0.0002`, weight decay `0.00001`, maximum three epochs, 16 interior PDE points,
and eight terminal points per surface batch. Record finite-loss, finite-gradient,
residual, and gradient-norm summaries. Success means numerical behavior only.

### Stage B frozen research run

Use all 7,500 train and 1,250 validation surfaces. Keep the 1,250 test surfaces
closed. Seeds are 11,22,33. Batch size is 32. AdamW learning rate is `0.0002`,
weight decay is `0.00001`, maximum epochs are 120, and patience is 15
evaluations. Select checkpoints solely by minimum validation total loss. Retain
all histories, hashes, logs, and environment evidence.

## 12. Compute plan and limitations

No Model 3 training runs locally. The pilot targets one Kaggle T4/P100 GPU
session below 8 GiB accelerator memory. The research run targets isolated
single-GPU sessions with persistent checkpoint snapshots. See
[MODEL3_PDE_CLOUD_EXECUTION_PLAN.md](MODEL3_PDE_CLOUD_EXECUTION_PLAN.md).

This is synthetic-data research, not market calibration. Explicit PDE physics
can regularize a learned solution but cannot manufacture identification absent
from ambiguous observations. Factor ordering is the declared tie-breaker for
factor swap symmetry.

## 13. Reproducibility requirements

A valid run records Git commit, dirty-state declaration, config SHA-256, dataset
SHA-256, Python/package versions, hardware, seeds, complete command, stdout and
stderr transcripts, checkpoint hashes, and metric manifests. Any identity
mismatch stops the run; it is never repaired by changing scientific settings.
Real Stage A additionally requires a clean tracked working tree before startup
and resume. Development smoke may run from a truthfully recorded dirty tree,
but is never represented as Stage A or a research result.
