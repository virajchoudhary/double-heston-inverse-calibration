# Canonical Double Heston PDE — Derivation and Physics Contract

Node C, overnight 2026-08-22. Branch `overnight/20260822-c-pde`, genesis
`642702e6706a3d17b3031619f35bda39bc144483`.

Everything below is derived from the repository's actual implementation, then
cross-checked against (i) an independent from-first-principles derivation by a
separate reviewer and (ii) numerical autograd experiments
(`tests_evidence/probe_residuals.py`, results in `probe_results.json`).

---

## 1. Stochastic specification as implemented (Phase A)

### 1.1 What the production pricer is

`src/double_heston.py` prices European calls by Gauss–Laguerre Fourier
inversion of the characteristic function of `log(S_T)`
(`price_double_heston_call`, 64-node rule). The characteristic function is

```
phi(u) = exp( iu(ln S + (r - q) T) + E_slow(u,T) + E_fast(u,T) )
```

with each factor's exponent (`heston_log_characteristic_exponent`,
src/double_heston.py:86-146) in the stable "Little Heston Trap" affine form

```
b   = kappa - rho*sigma*iu
d   = sqrt(b^2 + sigma^2 (u^2 + iu))        (Re d >= 0 branch)
g   = (b - d) / (b + d)
C   = (kappa*theta / sigma^2) [ (b - d) T - 2 ln( (1 - g e^{-dT}) / (1 - g) ) ]
D   = ((b - d) / sigma^2) * ( (1 - e^{-dT}) / (1 - g e^{-dT}) )
E_i = C_i + D_i * v0_i
```

`theta` enters only through `C` (linearly); `v0` only through `D` (linearly).
This is exactly the Heston-1993 affine exponent, applied per factor and SUMMED
in log-CF space (`src/double_heston.py:193-197`), i.e. the two factors'
characteristic functions MULTIPLY.

### 1.2 The SDE this represents (proven from the code structure)

A product/sum decomposition of this form exists if and only if the model is
the **affine two-spot-driver Double Heston** (Christoffersen–Heston–Jacobs
style): take four mutually independent standard Brownian motions
`B_slow, B_fast, Z_slow, Z_fast` and set

```
dW_i = rho_i dB_i + sqrt(1 - rho_i^2) dZ_i ,   i in {slow, fast}
```

Under the risk-neutral measure Q:

```
dS/S = (r - q) dt + sqrt(v_slow) dW_slow + sqrt(v_fast) dW_fast
dv_i  = kappa_i (theta_i - v_i) dt + sigma_i sqrt(v_i) dB_i ,   i = slow, fast
```

with `d<W_i, B_j> = rho_i dt` if `i = j` else `0`, and `d<B_slow, B_fast> = 0`.

Evidence chain (all proven from repository code):
- drift `(r - q)`: `src/double_heston.py:194` — the `iu(ln S + (r−q)T)` term.
- per-factor CIR/affine variance dynamics with per-factor leverage
  `rho_i` entering only via `b_i = kappa_i − rho_i sigma_i iu`: lines 121-146.
- factors combine ADDITIVELY in total spot variance (independent spot
  components): lines 161-162 docstring + additive log-CF decomposition.
- per-factor strict Feller gate `2 kappa theta − sigma^2 > 0`: lines 111-112,
  mirrored in the torch pricer (`src/torch_double_heston.py:311-312`) and in
  `src/constraints.py:13-15`.
- independent variance BMs: no `V_{v_slow,v_fast}`-generating covariance term
  exists anywhere in the CF; the repo's own `src/dheston/models/losses.py`
  residual likewise contains no `V_v1v2` term.

Numerical confirmation: the derived PDE below is satisfied by the differentiable
pricer to machine precision (rel. residual <= 1.3e-15 over 8 parameter/point
combinations, `probe_results.json: canonical_gl_pde_residual`), and the exact
additivity identity `E(u,T; theta, v0) = 2 E(u,T; theta/2, v0/2)` holds to
1.8e-15 (`factor_additivity_identity`) — the algebraic signature of independent
additive-variance factors.

### 1.3 What the correlation-disk constraint means

The repository enforces `rho_slow^2 + rho_fast^2 < 1`
(`src/constraints.py:39-46`, hard at pricing time; polar map in
`models/pinn_model.py` — see FINDINGS). For the implemented
4-Brownian construction the instantaneous covariance matrix of the state
`(S, v_slow, v_fast)` is

```
a = [ S^2(v_s+v_f)      rho_s sigma_s S v_s   rho_f sigma_f S v_f ]
    [ rho_s sigma_s S v_s   sigma_s^2 v_s          0            ]
    [ rho_f sigma_f S v_f        0             sigma_f^2 v_f    ]
```

with `det a = S^2 sigma_s^2 sigma_f^2 v_s v_f [ (1-rho_s^2) v_s + (1-rho_f^2) v_f ] >= 0`
whenever `|rho_i| < 1`. So the pointwise admissibility condition of the
IMPLEMENTED model is merely `|rho_i| < 1`.

`rho_slow^2 + rho_fast^2 < 1` is exactly the positive-definiteness condition of
the DIFFERENT, single-spot-driver 3-BM correlation matrix
`[[1, rho_s, rho_f], [rho_s, 1, 0], [rho_f, 0, 1]]` (det = 1 − rho_s^2 − rho_f^2).
For the implemented model the disk is therefore a **sufficient, conservative,
state-independent condition — not a necessary one** (e.g. rho = (−0.8, −0.7)
is pointwise admissible in the implemented model but excluded by the disk).
Classification: design choice, documented here; not an inconsistency. It
guarantees the observable instantaneous correlation matrix of
`(dS/S, dB_slow, dB_fast)` is uniformly PD for all variance states, since
`Corr(dS/S, dB_i) = rho_i sqrt(v_i / (v_s + v_f))` and
`rho_s^2 v_s + rho_f^2 v_f <= (rho_s^2 + rho_f^2)(v_s+v_f) < v_s + v_f`.

---

## 2. Derivation of the pricing PDE (Phase B)

Let `V(S, v_s, v_f, t)` be a European option value, `C^{1,2,2,2}` in the
interior. Itô multiplication table for the implemented model:

```
(dS)^2     = S^2 (v_s + v_f) dt
d<S, v_i>  = (S sqrt(v_i)) (sigma_i sqrt(v_i)) d<W_i, B_i> = rho_i sigma_i S v_i dt
(v_i diff)^2 = sigma_i^2 v_i dt
d<v_s, v_f> = sigma_s sigma_f sqrt(v_s v_f) d<B_s, B_f> = 0
```

Itô's formula applied to `e^{-rt} V` and the no-arbitrage (martingale)
condition give the backward PDE:

```
V_t + (r - q) S V_S
  + kappa_s (theta_s - v_s) V_vs + kappa_f (theta_f - v_f) V_vf
  + 1/2 (v_s + v_f) S^2 V_SS
  + 1/2 sigma_s^2 v_s V_vs vs + 1/2 sigma_f^2 v_f V_vf vf
  + rho_s sigma_s v_s S V_Svs + rho_f sigma_f v_f S V_Svf
  - r V = 0
```

Forward time-to-maturity form (`tau = T - t`, `U(S,v_s,v_f,tau) = V(...,T-t)`,
`U_tau = -V_t`):

```
U_tau = 1/2 (v_s + v_f) S^2 U_SS
      + 1/2 sigma_s^2 v_s U_vs vs + 1/2 sigma_f^2 v_f U_vf vf
      + rho_s sigma_s v_s S U_Svs + rho_f sigma_f v_f S U_Svf
      + (r - q) S U_S + kappa_s (theta_s - v_s) U_vs + kappa_f (theta_f - v_f) U_vf
      - r U
```

Key coefficient facts:
- cross-derivative coefficient is `rho_i sigma_i v_i S` (the two sqrt(v_i)
  factors from the spot loading and the variance diffusion multiply to `v_i`);
- **no `V_{vs vf}` term** — the variance Brownian motions are independent;
- off-diagonal Itô terms enter with the FULL covariation coefficient (each
  unordered pair appears once in the double sum with weight 1/2 x 2).

Cross-checks performed:
1. **Independent re-derivation** (separate reviewer, from the SDE only):
   term-by-term identical, including all coefficients and signs. The reviewer
   also proved the affine Riccati consistency: inserting
   `f = exp(A(tau) + B_s v_s + B_f v_f + u x)` yields two DECOUPLED Heston
   Riccati systems, possible only with the mixed coefficients `rho_i sigma_i v_i`
   (affine in the state) — an independent structural confirmation.
2. **Single-factor reduction**: killing factor `fast` (`theta_f = 0`,
   `v_f(0) = 0`) reproduces the classic Heston-1993 PDE term-by-term
   (all seven terms).
3. **Numerical**: the differentiable Gauss–Laguerre pricer satisfies the
   forward-tau PDE to ~1e-15 relative residual
   (`probe_results.json: canonical_gl_pde_residual`); deliberately corrupting
   one coefficient class raises the residual by 7-12 orders of magnitude
   (`rel_no_rho`, `rel_half_mix` columns), so the check is sensitive, not
   vacuous.
4. **Put-call parity**: `V_call - V_put = S e^{-q tau} - K e^{-r tau}` holds
   on the canonical pricer to 7e-15 (`put_call_parity`).

Note on evaluation through a Fourier pricer: by time-homogeneity, the pricer
output `P(S, v0_s, v0_f, tau)` equals `U(S, v_s = v0_s, v_f = v0_f, tau)`;
autograd derivatives w.r.t. `spot`, `tau`, and the `v0` entries therefore
evaluate the PDE operator exactly, PROVIDED the `v0` entries are genuine graph
leaves (see FINDINGS F2 for the Archive-2 failure mode).

---

## 3. Terminal and boundary conditions (Phase C)

Terminal (`tau = 0`, i.e. `t = T`), independent of variance state — implied by
the payoff structure and verified by the pricer's payoff at `maturity -> 0`:

```
call: U(S, v_s, v_f, 0) = max(S - K, 0)
put:  U(S, v_s, v_f, 0) = max(K - S, 0)
```

The repository's convention is European calls priced directly, puts by
put-call parity (`src/double_heston.py:290-319`) — parity verified to 7e-15.

Spot boundaries: `S = 0` is absorbing (`dS = S(...) = 0`): call -> 0,
put -> `K e^{-r tau}`. As `S -> infinity`, call ~ `S e^{-q tau} - K e^{-r tau}`,
put -> 0. (Standard; also the mathematical content of Archive-2's
`boundary_penalty` band, `src/dheston/models/losses.py:48-59`.)

Variance boundaries: `v_i = 0` is a degenerate (no boundary condition needed)
boundary of the square-root process: with the enforced strict Feller condition
`2 kappa_i theta_i > sigma_i^2` the factor is strictly positive a.s. and zero
is unattainable. No boundary condition in `v` is required or imposed by the
canonical pricer (Fourier solution). Archive-2's PDE loss imposes NO
terminal/boundary PDE conditions at all — its "boundary" term is the static
no-arbitrage band above, which the analytic pricer satisfies identically; it
is not boundary physics.

Limiting-case validations executed (Phase H):
- exact factor-additivity identity at CF level: error 1.8e-15;
- two identical half-factors == single Heston at price level, cross-validated
  against the independent Archive-2 COS single-factor pricer: max diff 3.2e-11
  on prices of size 4-15 (`one_factor_price_reduction`);
- deterministic-variance (sigma -> 0.02, rho -> -0.02) Black–Scholes limit:
  relative error <= 4.0e-4 across moneyness/maturities (`black_scholes_limit`),
  consistent with the O(sigma) convergence rate of the limit.

---

## 4. Assumption ledger

PROVEN from repository code (Phase A citations above):
- affine two-driver Double Heston construction; independent variance BMs;
  additive total spot variance; per-factor CIR dynamics; drift (r − q);
  per-factor leverage correlations; strict per-factor Feller enforcement;
  slow/fast kappa ordering enforcement; disk constraint enforcement.

INFERRED from mathematical structure (and confirmed numerically):
- the CF form is implementable in closed affine form ONLY under the two-driver
  construction (a single-driver construction with the same marginal dynamics
  would produce the non-affine cross-coefficient `rho_i sigma_i S sqrt(v_i(v_s+v_f))`
  and no closed CF of the implemented form exists);
- the disk constraint is sufficient-but-not-necessary pointwise (Section 1.3).

STILL UNCERTAIN / for mentor review:
- whether the research contract INTENDS the disk constraint as a modelling
  restriction (uniform instantaneous-correlation PD) or as a historical
  carry-over from a single-driver mental model — either is defensible, but the
  paper should state which;
- no time-dependent r(t), q(t) support; term structure is flat per surface.

External references used only to cross-check standard results (not copied):
Heston (1993) PDE form; Fang & Oosterlee (2008) COS truncation (Archive-2's
pricer); Christoffersen–Heston–Jacobs (2009) for the two-factor additive
variance construction. The PDE above was derived from the repository's SDE,
not transcribed from any paper.
