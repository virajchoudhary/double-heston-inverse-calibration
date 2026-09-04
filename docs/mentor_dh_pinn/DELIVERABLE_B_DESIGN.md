# Deliverable B — design document

The architecture built in Deliverable C. Every number quoted as "measured" here was measured
during the audit (Deliverable A) or while building this design, not assumed.

## B.1 The mapping

```
arbitrary set of option quotes
    -> permutation-invariant quote encoder
    -> ten parameter-query tokens, each cross-attending to the quotes
    -> location mu_z  and full covariance Sigma_z  (in latent coordinates)
    -> unrolled damped Gauss-Newton refinement against the EXACT Fourier pricer
    -> final parameters + uncertainty + identifiability + OOD status + true latency
```

One `forward()` call returns the calibration actually intended for use. There is no external
optimiser stage, so the reported latency is the latency of the answer.

## B.2 Why not one bigger MLP

Merging the two networks into a single pooled regressor would recreate the original failure.
Measured relative sensitivity `|d log C / d log p|` spans **30-300x** across the ten
parameters. On one 7-day at-the-money quote the audit measured

```
dC/dv0_slow  = 7.18e-2        dC/dkappa_slow = 1.32e-7        ratio 5.4e5
```

A single pooled vector feeding a dense layer lets the loud directions own the gradient. The
fix is not a hard short/long partition -- that was the previous architecture's brittleness --
but **ten separate information-extraction paths inside one model**: each parameter has its own
query token and its own cross-attention over the quotes, so `theta` and `kappa` can attend to
long maturities without competing with `v0` for the same pooled representation. Specialisation
becomes learned and soft rather than wired in.

## B.3 Quote encoder

Each quote is one token; strike and maturity are **data, not architecture**.

| stage | shape | note |
|---|---|---|
| per-quote features | `(B, N, 34)` | see below |
| linear embedding + noise embedding | `(B, N, 128)` | noise level added as a surface-level bias |
| 4 x pre-norm set-attention block | `(B, N, 128)` | key-padding masked |

Features (34) are scale-free: log-forward-moneyness `x = log(K/F)`, `log tau`, log normalised
price `C/(F e^{-r tau})`, log time value, normalised price, intrinsic, `r`, `q`, `sqrt(tau)`,
`x/sqrt(tau)`, plus 12 Fourier features of `x` and 12 of `log tau` for continuous positional
encoding. Nothing is indexed by strike number or expiry number.

`N` is whatever the market gives: 3, 9, 45, 92. Padding is a batching detail carried by an
explicit mask; **measured permutation invariance is 6.1e-16**, i.e. exact to floating point.

## B.4 Parameter tokens

Ten learned query tokens, one per Double Heston parameter. For each of `R = 3` rounds with
shared weights:

```
P <- P + CrossAttention(LN(P), X, X, key_padding_mask)
P <- P + SelfAttention(LN(P))
P <- P + FeedForward(LN(P))
```

Three rounds because the previous architecture measurably gained from iterated communication
(mutual awareness improved parameter RMSE by 7%, 0.1404 -> 0.1307). Here that becomes **all
ten parameter specialists communicating inside one model** rather than two networks passing
messages. Cross-attention weights are retained as an identifiability *diagnostic*, not as
evidence.

## B.5 Parameter coordinates: a bijection, not a box

`params_v2.decode : R^10 -> the engine's valid parameter set`, with an exact inverse.

| latent | map | parameter | bound |
|---|---|---|---|
| `z0` | softplus | `kappa_slow` | `(0, inf)` **unbounded** |
| `z1` | softplus | `kappa_fast - kappa_slow` | `(0, inf)` **unbounded**, gives the ordering |
| `z2` | exp | `theta_total` | `(0, inf)` **unbounded** |
| `z3` | share | `theta_slow / theta_total` | `(0, 1)` |
| `z4` | exp | `v0_total` | `(0, inf)` **unbounded** |
| `z5` | share | `v0_slow / v0_total` | `(0, 1)` |
| `z6`, `z7` | share | Feller ratios `eta_i = sigma_i / sqrt(2 kappa_i theta_i)` | `(0, 1)` |
| `z8`, `z9` | radial tanh | `(rho_slow, rho_fast)` | open unit disk, **surjective** |

Every bound is one the **engine itself enforces** (`src/constraints.py`), verified in the
audit: `price_double_heston_call` refuses a Feller-violating vector outright. There is no
`PARAM_BOX`. Measured:

* round trip is exact (2e-11) on vectors the old transform clipped in **5 of 10** coordinates;
* **0 invalid** in 90,000 draws at latent sd 3, 8 and 15;
* `encode` **raises** on an out-of-class vector instead of clamping.

Two subtleties the old design got wrong. `sigma` is reconstructed, not stored, so clipping
`theta` or `kappa` silently moved `sigma` as well. And `tanh` applied per component caps the
reachable correlation radius at `tanh(sqrt(2)) = 0.888`; the radius must come from the norm.

## B.6 Uncertainty head

```
mu_z    = Linear(token)            per token -> (B, 10)
diag    = softplus(Linear(token))  -> strictly positive Cholesky diagonal
offdiag = Linear([token_i ; token_j])  -> the 45 strictly-lower entries
Sigma_z = L L^T          L lower-triangular with that diagonal
```

Full covariance, not ten independent variances: Double Heston parameters are strongly
correlated and a diagonal head cannot represent the near-null directions that make the
problem ill-posed. Trained by full-covariance Gaussian negative log-likelihood in the same
latent coordinates the model works in. `L` is used directly in triangular solves; `Sigma`
is never inverted explicitly.

## B.7 The spring becomes a matrix

The audited stage two used a single global scalar, which had to be re-tuned per market
(`lambda = 3` synthetic, `lambda = 30` NIFTY -- a 10x swing). One scalar was being asked to
stand for quote noise, model mismatch, geometry quality, identifiability, and ten different
parameter directions at once.

It is replaced by the predicted covariance:

```
lambda ||p - mu||^2        ->        (z - mu_z)^T Sigma_z^-1 (z - mu_z)
```

Directions the data pins get a stiff prior; directions it does not get a loose one; and the
stiffness is per surface, per parameter, and per correlated direction.

## B.8 Differentiable physics refinement, inside forward()

`torch_pricer` is a port of the production engine, **not a surrogate**. Measured agreement
with `src/double_heston`: **8.4e-12** worst relative over 150 quotes (single path), **3.1e-15**
(batched). Autograd verified against 4th-order finite differences across low vol, high vol,
short T, long T, extreme rho, slow kappa, fast kappa, near-Feller, deep OTM and deep ITM:
worst **1.6e-8** relative to the largest derivative in the vector. (Relative to the *smallest*
derivative the finite-difference estimate is itself the unstable one -- it diverges as the
step shrinks, 5.6e-8 at h=1e-2 to 4.3e-3 at h=1e-7 -- so autograd is the accurate side.)

For `r = 0 .. R-1`, starting at `z_0 = mu_z`:

```
J   = d C(decode(z)) / dz            forward-mode, 10 tangents (N >> 10)
A   = J^T W J + Sigma_z^-1 + alpha I
g   = J^T W r + Sigma_z^-1 (z - mu_z)
z  <- z - clip(solve(A, g))          trust region, ||step|| <= 1.5
```

`W` is the mask divided by spot. The solve is 10x10; cost is dominated by exact pricing and
its Jacobian, never by the linear algebra. Non-finite prices or Jacobian rows are neutralised
explicitly and those surfaces are left where they are -- never silently zeroed.

## B.9 Training objective and curriculum

```
L = w_par L_parameter + w_unc L_uncertainty + w_phy L_clean_physics + w_ref L_refined
```

`L_parameter` Huber on `mu_z - z*`; `L_uncertainty` full-covariance Gaussian NLL;
`L_clean_physics` exact repricing of the pre-refinement parameters against **clean** prices;
`L_refined` the same after refinement, plus post-refinement parameter error. Inputs are
noisy; every target is clean, which is what teaches denoising rather than noise-chasing.

Each term is divided by a running EMA of itself before weighting, so the phase weights mean
what they say instead of being dominated by whichever term happens to be largest. Per-group
gradient norms at `mu_z` are logged every epoch; the acceptance test is that the `theta`/`kappa`
pathway is not orders of magnitude below `v0`. **Measured at epoch 1: theta/v0 = 1.17,
kappa/v0 = 1.19.**

| phase | epochs | input | refine | added objective |
|---|---|---|---|---|
| A clean amortised inference | 3 | clean | 0 | parameter + uncertainty |
| B noise and missing data | 4 | noisy | 0 | -- |
| C physics | 4 | noisy | 0 | exact clean repricing |
| D differentiable refinement | 3 | noisy | 2, backpropagated | post-refinement loss |

## B.10 Cost

Measured on this machine, float64 CPU.

| item | cost |
|---|---|
| data generation | 200,000 surfaces in **207 s** (old pipeline: ~78 min for a comparable corpus) |
| training step, batch 96 | **0.115 s** with length-bucketed batches (0.95 s without -- 8.3x) |
| epoch (833 batches + validation) | ~260 s |
| encoder forward, per surface | reported by the evaluation, not asserted here |
| each exact physics step | one exact price + a 10-tangent forward-mode Jacobian |

Length bucketing matters because the median surface has 28 quotes while the longest has 92;
padding every batch to the longest wasted roughly 3x the attention compute.
