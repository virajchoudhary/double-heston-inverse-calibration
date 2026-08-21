# Independent Cross-Derivation — Canonical Double Heston PDE

Provenance note (Node C, 2026-08-22): the derivation below was produced
INDEPENDENTLY from the identified SDE by a separate reviewer with no access
to Node C's derivation, then compared term-by-term. Result: identical PDE,
identical coefficients (rho_i*sigma_i*v_i*S cross terms, no V_v1v2),
identical tau-sign convention, plus an independent affine-Riccati-decoupling
structural argument. Committed so the evidence chain is self-contained.
Node C's own derivation lives in CANONICAL_DOUBLE_HESTON_PDE.md; the
adversarial review (which re-derived the generator a third time and
re-executed the experiments) is in
../tests_evidence/ADVERSARIAL_REVIEW_REPORT.md.

---

## 0. Setup: explicit Brownian construction

### 0.1 The construction used (the affine / standard "Christoffersen-style" Double Heston)

Take four mutually independent standard Brownian motions `(B_1, B_2, Z_1, Z_2)`
and define

```
dW_1 = rho_1 dB_1 + sqrt(1-rho_1^2) dZ_1
dW_2 = rho_2 dB_2 + sqrt(1-rho_2^2) dZ_2
```

Then `(W_1, W_2, B_1, B_2)` are jointly Gaussian BMs with instantaneous
correlation matrix (order W_1, W_2, B_1, B_2):

```
R_4 = [1   0   rho_1  0 ]      (block-diagonal after permuting to (W_1,B_1,W_2,B_2))
      [0   1   0   rho_2]      eigenvalues {1+-rho_1, 1+-rho_2}
      [rho_1  0   1   0 ]      PD <=> |rho_1|<1 and |rho_2|<1  (no disk constraint)
      [0   rho_2  0   1 ]
```

Model under Q:

```
dS/S = (r-q) dt + sqrt(v_1) dW_1 + sqrt(v_2) dW_2          (W_1 _|_ W_2, as stipulated)
dv_1  = kappa_1(theta_1-v_1) dt + sigma_1 sqrt(v_1) dB_1
dv_2  = kappa_2(theta_2-v_2) dt + sigma_2 sqrt(v_2) dB_2
```

So: `d<W_1,B_1> = rho_1 dt`, `d<W_2,B_2> = rho_2 dt`, all other cross-products
zero. The leverage correlation is attached to each factor's **own** spot
component. The *effective* spot-return/leverage correlation is then
state-dependent: `Corr(dS/S, dB_i) = rho_i*sqrt(v_i/(v_1+v_2))`.

### 0.2 The alternative "single spot driver" construction (for contrast — do not use)

The 3-BM construction — `dS/S = (r-q)dt + sqrt(v_1+v_2) dW_S` with
`d<W_S,B_i> = rho_i dt` — has the constant correlation matrix
`R_3 = [[1,rho_1,rho_2],[rho_1,1,0],[rho_2,0,1]]`, whose PD condition is
exactly `rho_1^2 + rho_2^2 < 1`. **However, this is a genuinely different
joint law**: it yields `d<S,v_i> = rho_i sigma_i S sqrt(v_i(v_1+v_2)) dt`,
which is non-affine. The two constructions agree on the marginal laws of S
and of (v_1,v_2) but differ precisely in the leverage coupling (the mixed
derivatives). The existence of the standard closed-form affine characteristic
function selects construction 0.1.

## 1. Ito multiplication table (construction 0.1)

Nonzero products (`dt*anything = 0`, `dt^2 = 0`):

```
(dW_1)^2 = dt,  (dW_2)^2 = dt,  (dB_1)^2 = dt,  (dB_2)^2 = dt
dW_1*dW_2 = 0                     (W_1 _|_ W_2)
dW_1*dB_1 = rho_1 dt,  dW_2*dB_2 = rho_2 dt
dW_1*dB_2 = 0,  dW_2*dB_1 = 0,  dB_1*dB_2 = 0
```

## 2. Quadratic covariations

**(i) d<S,S>:** `(dS)^2 = S^2(v_1 + v_2) dt` — total spot variance rate is
`S^2 (v_1+v_2)` (the cross `dW_1*dW_2` term vanishes).

**(ii) d<S,v_1>** — only the `sqrt(v_1) dW_1` part of dS can covary with `dB_1`:

```
d<S,v_1> = (S sqrt(v_1)) (sigma_1 sqrt(v_1)) d<W_1,B_1> = S sigma_1 v_1 rho_1 dt
```

Key point: **sqrt(v_1) * sqrt(v_1) = v_1**. Similarly
`d<S,v_2> = rho_2 sigma_2 S v_2 dt`.

**(iii) d<v_1,v_2> = sigma_1 sigma_2 sqrt(v_1 v_2) d<B_1,B_2> = 0.**

**(iv) d<v_i,v_i> = sigma_i^2 v_i dt.**

Instantaneous covariance-rate matrix of the state X = (S, v_1, v_2):

```
        [ S^2(v_1+v_2)    rho_1 sigma_1 S v_1      rho_2 sigma_2 S v_2 ]
a   =   [ rho_1 sigma_1 S v_1     sigma_1^2 v_1        0            ]
        [ rho_2 sigma_2 S v_2        0             sigma_2^2 v_2   ]
```

PSD check: `det a = S^2 sigma_1^2 sigma_2^2 v_1 v_2 [(1-rho_1^2) v_1 + (1-rho_2^2) v_2] >= 0`
whenever `|rho_i| <= 1` (the pointwise admissibility condition).

## 3. Ito expansion and the backward PDE

For `V(S,v_1,v_2,t)` in `C^{1,2,2,2}`:

```
dV = V_t dt + V_S dS + V_{v_1}dv_1 + V_{v_2}dv_2
   + 1/2 V_{SS}d<S,S> + 1/2 V_{v_1v_1}d<v_1,v_1> + 1/2 V_{v_2v_2}d<v_2,v_2>
   + V_{Sv_1}d<S,v_1> + V_{Sv_2}d<S,v_2> + V_{v_1v_2}d<v_1,v_2>
```

(each off-diagonal pair appears twice with coefficient 1/2, i.e. **once with
the full covariation** — this is why the mixed-derivative coefficient is
`1*rho_i sigma_i v_i S`, not `1/2*rho_i sigma_i v_i S` nor `2*rho_i sigma_i v_i S`).

**Feynman-Kac / no-arbitrage step.** The arbitrage-free price is
`V(t,X_t) = E^Q[e^{-r(T-t)} h(S_T) | X_t]`, hence `M_s = e^{-rs}V(s,X_s)` is a
Q-martingale. Applying Ito to `e^{-rt}V`:

```
d(e^{-rt}V) = e^{-rt}[ (V_t + LV - rV) dt ] + d(martingale)
```

A continuous martingale cannot have a nonzero finite-variation drift part, so
`V_t + LV - rV = 0` identically.

**Backward PDE:**

```
V_t + (r-q) S V_S + kappa_1(theta_1-v_1) V_{v_1} + kappa_2(theta_2-v_2) V_{v_2}
    + 1/2 (v_1+v_2) S^2 V_{SS} + 1/2 sigma_1^2 v_1 V_{v_1v_1} + 1/2 sigma_2^2 v_2 V_{v_2v_2}
    + rho_1 sigma_1 v_1 S V_{Sv_1} + rho_2 sigma_2 v_2 S V_{Sv_2}
    - r V = 0
```

In log-spot `x = ln S` (via `d<x,v_i> = rho_i sigma_i v_i dt`,
`d<x,x> = (v_1+v_2)dt`, Ito drift `r-q-1/2(v_1+v_2)`):

```
V_t + [r - q - 1/2(v_1+v_2)] V_x + sum_i kappa_i(theta_i-v_i) V_{v_i}
    + 1/2(v_1+v_2) V_{xx} + 1/2 sigma_1^2 v_1 V_{v_1v_1} + 1/2 sigma_2^2 v_2 V_{v_2v_2}
    + rho_1 sigma_1 v_1 V_{xv_1} + rho_2 sigma_2 v_2 V_{xv_2} - rV = 0
```

(Every coefficient is affine in (v_1,v_2) — the model is in the affine class.)

## 4. Forward time-to-maturity form

Set `tau = T - t >= 0` and `U(S,v_1,v_2,tau) := V(S,v_1,v_2,T-tau)`. Then
`U_tau = -V_t`, so

```
U_tau = 1/2(v_1+v_2) S^2 U_{SS} + 1/2 sigma_1^2 v_1 U_{v_1v_1} + 1/2 sigma_2^2 v_2 U_{v_2v_2}
      + rho_1 sigma_1 v_1 S U_{Sv_1} + rho_2 sigma_2 v_2 S U_{Sv_2}
      + (r-q) S U_S + kappa_1(theta_1-v_1) U_{v_1} + kappa_2(theta_2-v_2) U_{v_2}
      - r U
```

i.e. `U_tau = LU - rU`, an initial-value (parabolic, forward) problem. The
sign flip `V_t -> -U_tau` is the only change; all spatial terms are identical.

## 5. Terminal conditions

At `t = T` (equivalently `tau = 0`), independent of the variance factors:

```
Call:  V(S,v_1,v_2,T) = max(S - K, 0)
Put:   V(S,v_1,v_2,T) = max(K - S, 0)
```

Boundary behavior: `S = 0` is absorbing (`dS = S*(*) = 0`), giving call = 0,
put = `K e^{-r tau}`; as `S -> infinity`, call ~ `S e^{-q tau} - K e^{-r tau}`,
put -> 0. At `v_i = 0` the PDE is degenerate and **no** boundary condition is
imposed.

## 6. Sanity checks

### (a) Single-factor Heston reduction

Kill factor 2 by setting `theta_2 = 0` **and** `v_2(0) = 0`. Then at `v_2 = 0`:
drift `kappa_2(0-0) = 0`, diffusion `sigma_2 sqrt(0) = 0`, so `v_2 == 0` is
the (unique strong) solution. (`v_2(0)=0` alone does *not* work — the drift
`kappa_2 theta_2 > 0` immediately re-inflates it; `theta_2=0` alone only gives
mean reversion *toward* zero.) With `v_2 == 0`, every factor-2 term vanishes
and the SDE reduces to a single-driver Heston. Term-by-term against Heston
(1993): all seven terms match exactly — time `V_t`; spot drift `(r-q)SV_S`;
spot diffusion `1/2 v_1 S^2 V_{SS}`; variance drift
`kappa_1(theta_1-v_1)V_{v_1}`; mixed `rho_1 sigma_1 v_1 S V_{Sv_1}`; variance
diffusion `1/2 sigma_1^2 v_1 V_{v_1v_1}`; discount `-rV`.

### (b) Mixed-derivative coefficient is `rho_i sigma_i v_i S`

From Section 2(ii): the spot's loading on the Brownian that drives `v_i` is
`S sqrt(v_i) * rho_i` and the variance loading is `sigma_i sqrt(v_i)`; the
Ito product gives `S sigma_i rho_i * (sqrt(v_i)*sqrt(v_i)) = rho_i sigma_i v_i S`.
Error taxonomy:
- `rho_i sigma_i sqrt(v_i) S` — drops one sqrt(v_i) factor;
- `1/2 rho_i sigma_i v_i S` or `2 rho_i sigma_i v_i S` — wrong factor-of-2
  bookkeeping in the symmetric Ito double sum;
- `rho_i sigma_i S sqrt(v_i(v_1+v_2))` — the single-driver construction;
  non-affine; not the standard Double Heston;
- in log coordinates the coefficient is `rho_i sigma_i v_i` (no `S`).

### (c) No `V_{v_1v_2}` term

Its coefficient would be `d<v_1,v_2>/dt = sigma_1 sigma_2 sqrt(v_1 v_2) *
Corr(dB_1,dB_2) = 0`. Structural reason: quadratic covariation of two Ito
processes depends **only** on the instantaneous covariance of their driving
Brownian motions; since `B_1 _|_ B_2`, the covariation vanishes identically.
`(v_1)` and `(v_2)` are not merely uncorrelated but **independent** processes.
The two factors are coupled only *additively*, through `1/2(v_1+v_2)S^2V_{SS}`
(and the log-drift) — exactly why the Double Heston characteristic function
decomposes into two Heston-type Riccati systems glued through the spot terms.

### (d) Bonus checks

- **Put-call parity is a solution:** `F = Se^{-q tau} - Ke^{-r tau}` gives
  `F_t + (r-q)SF_S - rF = [q + (r-q) - r]Se^{-q tau} + [-r + r]Ke^{-r tau} = 0`.
- **Affineness / Riccati consistency:** with
  `f = exp(A(tau) + B_1(tau)v_1 + B_2(tau)v_2 + ux)` inserted into
  `f_tau = Lf`, the ODEs decouple across factors:
  `B_i' = 1/2(u^2-u) + (u rho_i sigma_i - kappa_i)B_i + 1/2 sigma_i^2 B_i^2`,
  `A' = (r-q)u + sum_i kappa_i theta_i B_i` — two independent Heston Riccati
  equations, possible **only** because the mixed coefficients are
  `rho_i sigma_i v_i` (affine). This independently confirms the PDE's
  second-order coefficients.
- **Covariance matrix `a` is PSD** per the determinant expression in Section 2.

## 7. Subtleties

**(1) The rho-disk question.** For
`R_3 = [[1,rho_1,rho_2],[rho_1,1,0],[rho_2,0,1]]`: `det R_3 = 1 - rho_1^2 - rho_2^2`;
eigenvalues `{1, 1-sqrt(rho_1^2+rho_2^2), 1+sqrt(rho_1^2+rho_2^2)}`. By
Sylvester's criterion, `R_3 > 0 <=> rho_1^2 + rho_2^2 < 1` — the disk
condition is exactly the PD condition of that 3x3 matrix. Caveat: that
constant 3x3 matrix belongs to the *single-spot-driver* construction
(Section 0.2). In the two-driver construction actually used, the Brownian
correlation matrix is the 4x4 `R_4`, block-diagonal in the pairs `(W_i,B_i)`,
with eigenvalues `{1+-rho_1, 1+-rho_2}` — PD <=> `|rho_1|<1 and |rho_2|<1`,
with **no** disk constraint. Reconciliation: the observable instantaneous
correlation matrix of `(dS/S, dB_1, dB_2)` in the two-driver model is
state-dependent, with `Corr(dS/S,dB_i) = rho_i sqrt(v_i/(v_1+v_2))`, and it
is PD iff `rho_1^2 v_1 + rho_2^2 v_2 < v_1 + v_2` — automatically satisfied
when `|rho_i|<1`. The uniform condition `rho_1^2+rho_2^2<1` is a sufficient,
state-independent condition (and the necessary one for a single spot driver).

**(2) Which construction.** Constant `Corr(dS/S, dB_i) = rho_i` and the mixed
coefficient `rho_i sigma_i v_i S` are mutually exclusive — the former forces
`rho_i sigma_i S sqrt(v_i(v_1+v_2))`. Affineness (and the classic limit)
selects the two-driver construction.

**(3) Sign conventions.** (i) Backward: `V_t + LV - rV = 0` with terminal
payoff; forward: `U_tau = LV - rU` with initial payoff. (ii) rho_i defined
via `d<W_i,B_i> = rho_i dt`; the mixed term enters with sign
`+rho_i sigma_i v_i S V_{Sv_i}` (equity skew typically rho_i < 0). (iii)
Discount `-rV`; dividend only through the drift, giving `+(r-q)SV_S`.

**(4) Boundary/degeneracy.** `v_i = 0` is a degenerate boundary: no boundary
condition required (if Feller `2 kappa_i theta_i >= sigma_i^2`, zero is
unattainable; otherwise instantaneously reflecting). `S = 0` is absorbing.
Degenerate parabolic on the coordinate planes, uniformly parabolic in the
interior; classical solutions in the interior, viscosity arguments at the
boundary.

**(5) Time-homogeneity.** Constant r, q assumed; for deterministic
time-dependent rates replace by term averages over (t,T].

## 8. Final PDEs (clean statement)

**Backward (calendar time t):**

```
V_t + (r-q) S V_S + kappa_1(theta_1-v_1) V_{v_1} + kappa_2(theta_2-v_2) V_{v_2}
    + 1/2 (v_1+v_2) S^2 V_SS
    + 1/2 sigma_1^2 v_1 V_{v_1v_1} + 1/2 sigma_2^2 v_2 V_{v_2v_2}
    + rho_1 sigma_1 v_1 S V_{Sv_1} + rho_2 sigma_2 v_2 S V_{Sv_2}
    - r V = 0
```

with `V(S,v_1,v_2,T) = (S-K)^+` (call) or `(K-S)^+` (put).

**Forward (time to maturity tau = T-t):**

```
U_tau = 1/2 (v_1+v_2) S^2 U_SS
      + 1/2 sigma_1^2 v_1 U_{v_1v_1} + 1/2 sigma_2^2 v_2 U_{v_2v_2}
      + rho_1 sigma_1 v_1 S U_{Sv_1} + rho_2 sigma_2 v_2 S U_{Sv_2}
      + (r-q) S U_S + kappa_1(theta_1-v_1) U_{v_1} + kappa_2(theta_2-v_2) U_{v_2}
      - r U
```

with `U(S,v_1,v_2,0) = (S-K)^+` or `(K-S)^+`.

Key covariations used: `d<S,S> = S^2(v_1+v_2)dt`, `d<S,v_i> = rho_i sigma_i S v_i dt`,
`d<v_i,v_i> = sigma_i^2 v_i dt`, `d<v_1,v_2> = 0` — hence mixed coefficients
`rho_i sigma_i v_i S` on `V_{Sv_i}` and **no** `V_{v_1v_2}` term. Reduction to
Heston (1993) is exact term-by-term; put-call parity and the affine Riccati
structure were verified as independent cross-checks.
