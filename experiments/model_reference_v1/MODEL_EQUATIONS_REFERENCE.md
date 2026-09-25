# Model equations, conditions, parameters and data ranges

Every formula below was read out of the code, with the file and function named. Nothing is quoted
from memory.

Notation used throughout:

| symbol | meaning |
|---|---|
| $S$ | underlying price |
| $K$ | strike |
| $F$ | forward, $F = S e^{(r-q)\tau}$ |
| $\tau = T-t$ | remaining time to maturity, in years |
| $x = \log(F/K)$ | log-moneyness (positive = low strike) |
| $c = C/F$ | forward-normalised call price (undiscounted) |
| $w = \sigma_{imp}^2 \tau$ | total implied variance |
| $v_s, v_f$ | instantaneous variance of the slow and fast factors |

---

# PART 1 — THE THREE PRICING MODELS

## 1.1 Black-Scholes / Black-76

Implemented in `experiments/btc_multifactor_v1/engine.py::black`, `vega`, `iv`.

**Price** (forward-normalised, undiscounted):

$$c(x,\tau,\sigma) = \Phi(d_1) - e^{-x}\,\Phi(d_2), \qquad
d_1 = \frac{x}{\sigma\sqrt{\tau}} + \frac{\sigma\sqrt{\tau}}{2}, \qquad d_2 = d_1 - \sigma\sqrt{\tau}$$

In currency units $C = F\,c$, and with $r=q=0$ (the controlled benchmark) $F=S$ so $C = S\,c$.

**Vega** (used as the calibration weight):

$$\nu(x,\tau,\sigma) = \frac{\partial c}{\partial \sigma} = \varphi(d_1)\sqrt{\tau},
\qquad \varphi(z)=\tfrac{1}{\sqrt{2\pi}}e^{-z^2/2}$$

**Implied volatility.** `engine.iv` converts to the $C/K$ form and inverts
`regular_pinn_data.black_call(x,w) = e^{x}\Phi(d)-\Phi(d-\sqrt{w})`, $d = x/\sqrt{w}+\sqrt{w}/2$,
by 75 bisection steps on $w\in[10^{-12},\,40]$, then $\sigma_{imp}=\sqrt{w/\tau}$.

**Inversion validity condition** (`invert_total_variance`) — outside it the implied vol is returned
as NaN, which is what produces the gaps in deep out-of-the-money short-dated smiles:

$$\max(e^{x}-1,\,0) + 10^{-12} \;<\; C/K \;<\; e^{x} - 10^{-12}$$

**Two Black-Scholes variants are used as baselines:**

- `BS_FLAT` — a single $\sigma$ for the whole surface.
- `BS_TERM` / `BS_EXPIRY` — one volatility $\sigma_j$ per calibration expiry, with
  $$w(\tau) = \text{linear interpolation of } \tau_j\sigma_j^2 \text{ in } \tau, \qquad
  \sigma(\tau)=\sqrt{w(\tau)/\tau}$$
  and flat extrapolation beyond the first and last knots (`engine.bs_predict`,
  `amend01.predict_bs_expiry`). This is the strongest non-leaking Black-Scholes in the project.

## 1.2 Single Heston

$$dS_t = \sqrt{v_t}\,S_t\,dW_t^S, \qquad
dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v, \qquad
d\langle W^S, W^v\rangle_t = \rho\,dt$$

Five parameters $(\kappa,\theta,\sigma,\rho,v_0)$. Priced by the same Fourier machinery as Double
Heston with a single factor (`engine.Grid` accepts a 5-vector).

## 1.3 Double Heston — separable four-shock convention

$$dS_t = \sqrt{v_{s,t}+v_{f,t}}\;S_t\,dW^S_t$$
$$dv_{s,t} = \kappa_s(\theta_s - v_{s,t})dt + \sigma_s\sqrt{v_{s,t}}\,dW^{s}_t, \qquad
dv_{f,t} = \kappa_f(\theta_f - v_{f,t})dt + \sigma_f\sqrt{v_{f,t}}\,dW^{f}_t$$

with **four separate shocks**: the price shock decomposes so that each variance factor has its own
correlation $\rho_s$, $\rho_f$ with the price, and the two variance factors are independent of each
other. The instantaneous total variance is $v_s+v_f$.

**Characteristic exponent per factor** (`src/double_heston_reference.py::_factor_exponent`,
Little-Heston-Trap stable form). For one factor with $(\kappa,\theta,\sigma,\rho,v_0)$:

$$b = \kappa - \rho\sigma\,iu, \qquad
d = \sqrt{b^2 + \sigma^2\,(u^2 + iu)}\;\;(\text{branch: } \Re d \ge 0), \qquad
g = \frac{b-d}{b+d}$$

$$\psi(u,\tau) = \frac{\kappa\theta}{\sigma^2}\Big[(b-d)\tau - 2\log\frac{1-g e^{-d\tau}}{1-g}\Big]
\;+\; \frac{b-d}{\sigma^2}\cdot\frac{1-e^{-d\tau}}{1-g e^{-d\tau}}\;v_0$$

**The two factors multiply**, which is the whole point of the separable construction:

$$\tilde\varphi(u,\tau) = \exp\big[\psi_s(u,\tau) + \psi_f(u,\tau)\big]$$

**Pricing integral** (`engine.Grid.__call__`), forward-normalised and undiscounted:

$$c(x,\tau) = \underbrace{\frac{1}{2} + \frac{1}{\pi}\int_0^\infty
\Re\Big[\frac{e^{iux}\,\tilde\varphi(u-i,\tau)}{iu}\Big]du}_{P_1}
\;-\; e^{-x}\underbrace{\Big(\frac{1}{2} + \frac{1}{\pi}\int_0^\infty
\Re\Big[\frac{e^{iux}\,\tilde\varphi(u,\tau)}{iu}\Big]du\Big)}_{P_2}$$

Evaluated by **Gauss-Laguerre quadrature** with weights $w_k e^{u_k}/(\pi i u_k)$ at 128 nodes and
again at 96 nodes.

---

# PART 2 — CONDITIONS THAT MUST BE SATISFIED

## 2.1 Parameter admissibility (`engine.admissible`, `src/constraints.py`)

| condition | formula | where enforced |
|---|---|---|
| positivity | $\kappa_i,\theta_i,\sigma_i,v_{0,i} > 0$ | always |
| correlation bounds | $-1 < \rho_i < 1$ | always |
| slow-first ordering | $\kappa_s < \kappa_f$ | always (identifiability: removes the label swap) |
| **Feller (strict variant)** | $2\kappa_i\theta_i - \sigma_i^2 > 0$ | `feller_strict` only |
| **joint correlation disk** | $\rho_s^2 + \rho_f^2 < 1$ | repository reference contract only, **not** the market tests |

The market experiments use the `feller_free` variant with individual $|\rho_i|<1$ and no joint disk:
the characteristic function stays valid without Feller, and the validation dates chose `feller_free`
for both Heston models (median held-out IV RMSE 3.81 vs 6.88 for Double Heston).

Under the **strict** variant the vol-of-vol is parameterised to satisfy Feller by construction:
$\sigma_i = \eta_i\sqrt{2\kappa_i\theta_i}$ with $\eta_i\in[0.01,\,0.999]$.

## 2.2 Numerical acceptance (`engine.exact`)

A price is accepted only if all of these hold, otherwise the point is recomputed with independent
adaptive quadrature and the fallback is logged:

$$|c_{128} - c_{96}| \le 10^{-7}, \qquad c \ge \max(1-e^{-x},0) - 10^{-9}, \qquad c \le 1 + 10^{-9}$$

## 2.3 No-arbitrage conditions checked on every plotted curve

$$\frac{\partial C}{\partial S}\in[0,1], \qquad \frac{\partial^2 C}{\partial S^2}\ge 0, \qquad
\frac{\partial^2 C}{\partial K^2}\ge 0, \qquad \frac{\partial w}{\partial \tau}\ge 0, \qquad
\max(S-K,0)\le C\le S$$

and $C\to\max(S-K,0)$ as $\tau\to 0$. Result across all 30 controlled curves: **zero violations**
(`clear_separation_figures_v1/data/controlled_shape_checks.json`).

## 2.4 Statistical conditions for a claimed win (frozen endpoint)

Double Heston "beats" a competitor only if **both** hold on held-out quotes:

$$p_{\text{Wilcoxon, one-sided}} < 0.05 \qquad \text{and} \qquad
\text{lower bound of the 95\% cluster bootstrap of } \overline{\Delta} > 0$$

where $\Delta = \text{RMSE}_{\text{competitor}} - \text{RMSE}_{DH}$ averaged within each cluster
(shock episode, or calendar week), 5,000 replicates, seed 20260920.

---

# PART 3 — THE PINN

## 3.1 What the network actually outputs

The network never predicts a price directly. It predicts a **bounded multiplicative correction** to
an analytic baseline volatility (`src/mentor_dh_pinn/regular_pinn_torch.py`).

**Analytic baseline — expected average variance over the option's life:**

$$\bar v(\tau) \;=\; \sum_{i\in\{s,f\}}\Big[\theta_i + (v_i-\theta_i)\,
\frac{1-e^{-\kappa_i\tau}}{\kappa_i\tau}\Big]$$

**Bounded correction and output:**

$$\mathrm{corr} = 1.8\cdot\tanh\big(\mathrm{head}(h)\big), \qquad
\sigma_{imp} = \sqrt{\bar v}\;e^{\mathrm{corr}}, \qquad
\ell \equiv \log w = \log\tau + \log\bar v + 2\,\mathrm{corr}$$

$$C/K = \mathrm{black\_call}(x, w) = e^{x}\Phi(d) - \Phi(d-\sqrt{w}), \quad d = \frac{x}{\sqrt w}+\frac{\sqrt w}{2}$$

Two consequences follow **by construction**, not by training: the payoff at $\tau=0$ is exact, and
the price cannot leave its no-arbitrage bounds.

## 3.2 The 24 engineered features (fixed formulas, nothing learned)

With $z = x/\sqrt{\tau\bar v + x^2/64}$ and $\eta_i = \sigma_i/\sqrt{2\kappa_i\theta_i}$:

**4 general**
$$\frac{x}{0.36},\qquad \frac{z}{8},\qquad \tanh\!\big(z/1.5\big),\qquad
\frac{2\log(\tau/\tau_{\min})}{\log(\tau_{\max}/\tau_{\min})}-1$$

**9 per factor** $i\in\{s,f\}$ (18 total), each $\log$ mapped to $[-1,1]$ over the stated range:
$$\log v_i\,[0.01,0.3],\quad \log\kappa_i\,[0.15,12],\quad \log\theta_i\,[0.01,0.3],\quad \rho_i,$$
$$\tanh(\kappa_i\tau),\quad \nu_i=\tanh\!\Big(\frac{\sigma_i\sqrt\tau}{\sqrt{\bar v}}\Big),\quad
\rho_i\nu_i,\quad \tanh\!\Big(\tfrac12\log\frac{v_i}{\theta_i}\Big),\quad
\frac{2\log(1+\eta_i)}{\log 4}-1$$

**2 cross-factor**
$$\frac{2v_s}{v_s+v_f}-1, \qquad \tanh(z/1.5)\cdot(\rho_s\nu_s+\rho_f\nu_f)$$

## 3.3 The PINN loss function

Per-term definitions (`dh_pinn_v5/train.py::parts_of`):

$$L_{price} = \Big\langle \big(\hat c_\theta - c^{teacher}\big)^2\Big\rangle, \qquad
L_{IV} = \Big\langle \big(\hat\sigma_\theta - \sigma^{teacher}\big)^2\Big\rangle_{\text{valid IV only}}$$

$$L_{PDE} = \big\langle \mathcal{R}^2 \big\rangle, \qquad
L_{conv} = \big\langle \max(-\mathcal{C},0)^2 \big\rangle$$

where $\mathcal{R}$ is the scaled two-factor Heston PDE residual and $\mathcal{C}$ the convexity
diagnostic, both from `regular_pinn_torch.residual`, written in log-total-variance form
$\ell=\log w$ and normalised by $\text{scale} = (v_s+v_f) + \bar v$:

$$\mathcal{R} \;=\; \frac{1}{\text{scale}}\Big[
w\,\ell_\tau
- \tfrac12 (v_s+v_f)\,\mathcal{D}_x
- \sum_i \rho_i\sigma_i v_i\,\mathcal{D}_{xv_i}
- \tfrac12\sum_i \sigma_i^2 v_i\,\mathcal{D}_{v_iv_i}
- \sum_i \kappa_i(\theta_i - v_i)\,w\,\ell_{v_i}\Big]$$

with, writing $a = \tfrac12 x^2 - \tfrac18 w^2 - \tfrac12 w$,

$$\mathcal{D}_x = 2 + 2(\tfrac{w}{2}-x)\ell_x + a\,\ell_x^2 + w(\ell_{xx}+\ell_x^2) - w\,\ell_x$$
$$\mathcal{D}_{xv_i} = (\tfrac{w}{2}-x)\ell_{v_i} + a\,\ell_x\ell_{v_i} + w(\ell_{xv_i}+\ell_x\ell_{v_i})$$
$$\mathcal{D}_{v_iv_i} = a\,\ell_{v_i}^2 + w(\ell_{v_iv_i}+\ell_{v_i}^2)$$
$$\mathcal{C} = \Big(1-\tfrac12 x\ell_x\Big)^2 - \tfrac14 w\ell_x^2 - \tfrac{w^2\ell_x^2}{16}
+ \tfrac12 w(\ell_{xx}+\ell_x^2)$$

All derivatives are taken by automatic differentiation with `create_graph=True`.

**Total loss** — every supervised term is divided by the square of its own *acceptance tolerance*,
which is the v4 fix that cured the earlier underfitting:

$$\boxed{\;L \;=\; \frac{L_{price}}{(2\times10^{-5})^2}
\;+\; w_{IV}\,\frac{L_{IV}}{(0.002)^2}
\;+\; w_{PDE}\cdot 0.1\cdot\frac{L_{PDE}}{(0.01)^2}
\;+\; w_{conv}\cdot 0.1\cdot L_{conv}\;}$$

In the **locked v4 network** $w_{IV}=w_{PDE}=w_{conv}=1$ (fixed).
In the **improved v5 network** they are updated every 100 steps by gradient-norm balancing
(Wang, Teng & Perdikaris learning-rate annealing):

$$w_i \;\leftarrow\; 0.9\,w_i + 0.1\cdot\min\!\Big(\frac{\|\nabla_\theta L_{price}\|}{\|\nabla_\theta L_i\|},\,10^{3}\Big)$$

**Acceptance gates** (frozen before training, all four must pass on the development split):

$$\text{price RMSE}\le 2\times10^{-5},\quad P_{95}\le 5\times10^{-5},\quad
\max\le 2\times10^{-4},\quad \text{IV RMSE}\le 0.002\ (=0.2\ \text{vol pts})$$

## 3.4 Training configuration

| | locked v4 | improved v5 |
|---|---|---|
| body | plain tanh stack, width 256, depth 5 | gated blocks, width 256, depth 5 |
| parameters | 269,825 | 282,632 |
| teacher points | 100,000 exact Fourier prices | same |
| collocation points | 18,000 fixed | 18,000, resampled every 2,000 steps (RAD) |
| optimiser | Adam 40,000 steps, $10^{-3}\to10^{-5}$ cosine, then L-BFGS 300 | same |
| batch | 512 labels / 128 physics | same |
| seeds | 17, 43 (prices averaged) | same |
| loss weights | fixed | gradient-norm balanced |
| activation | $\tanh$ | $\tanh(a_\ell z)$, $a_\ell$ learned per block |

**Residual-adaptive sampling (RAD, v5 only).** Every 2,000 steps, 40,000 candidates are drawn from a
120,000-point pool, their $|\mathcal{R}|$ is evaluated in chunks, and half of the 18,000 active points
are redrawn without replacement with probability

$$p_j \;\propto\; \frac{|\mathcal{R}_j|}{\overline{|\mathcal{R}|}} + 1$$

the other half staying uniform. The budget never increases.

---

# PART 4 — THE HARD-CODED PARAMETERS

Source: Christoffersen, Heston & Jacobs (2009); stored in
`experiments/nifty_multifactor_v4/config.json` as `published_double_slow_first` and
`published_single`.

## 4.1 The ten Double Heston parameters (slow factor first)

| # | symbol | value | meaning |
|---|---|---|---|
| 1 | $\kappa_s$ | **0.9491** | slow mean-reversion speed (half-life ≈ 0.73 y) |
| 2 | $\theta_s$ | **0.0257** | slow long-run variance (16.0% vol) |
| 3 | $\sigma_s$ | **0.0517** | slow vol-of-vol |
| 4 | $\rho_s$ | **+0.7009** | slow price/variance correlation |
| 5 | $v_{s,0}$ | **0.0003** | slow initial variance (1.7% vol) |
| 6 | $\kappa_f$ | **10.7526** | fast mean-reversion speed (half-life ≈ 24 days) |
| 7 | $\theta_f$ | **0.0330** | fast long-run variance (18.2% vol) |
| 8 | $\sigma_f$ | **0.3613** | fast vol-of-vol |
| 9 | $\rho_f$ | **−0.8916** | fast price/variance correlation |
| 10 | $v_{f,0}$ | **0.0252** | fast initial variance (15.9% vol) |

Key structural facts: $\kappa_f/\kappa_s = 11.33$ (two clearly separated timescales), and
$\rho_s>0>\rho_f$ — **opposite-signed correlations**, which is the mechanism behind the short-dated
smirk flexibility and the reason the separable four-shock convention is required.

Derived quantities used throughout:
$$v_{s,0}+v_{f,0} = 0.0255 \;\Rightarrow\; \sigma_{BS,\text{fixed}} = \sqrt{0.0255} = 15.97\%$$
$$\theta_s+\theta_f = 0.0587 \;\Rightarrow\; \text{long-run total volatility } 24.2\%$$

**Feller check:** slow $2\kappa_s\theta_s-\sigma_s^2 = 0.0461 > 0$ ✓;
fast $2\kappa_f\theta_f-\sigma_f^2 = 0.5791 > 0$ ✓. **Joint disk:** $\rho_s^2+\rho_f^2 = 1.286 > 1$,
so the published set satisfies the individual bounds but **not** the repository's stricter joint-disk
convention — which is why the market tests use the individual-$\rho$ contract.

## 4.2 The five Single Heston parameters (published)

$$(\kappa,\theta,\sigma,\rho,v_0) = (8.9814,\;0.0409,\;0.2970,\;-0.9621,\;0.0244)$$

Feller: $2\kappa\theta-\sigma^2 = 0.6465>0$ ✓.

## 4.3 The level scale $s$ (the one number ever fitted to fixed-parameter models)

$$\theta_i \mapsto s\,\theta_i, \qquad \sigma_i \mapsto \sqrt{s}\,\sigma_i, \qquad
v_{i,0}\mapsto s\,v_{i,0}, \qquad \sigma_{BS}\mapsto \sqrt{s}\,\sigma_{BS}, \qquad s\in[0.7,\,3.2]$$

This rescales the volatility **level** while leaving every timescale, correlation and shape untouched.
The range is exactly the range the PINN was trained on.

## 4.4 The controlled two-timescale scenario `FIXED_TOTAL_TWIST`

Same published parameters, only the two initial variances swapped:

| state | $v_{f,0}$ | $v_{s,0}$ | total | instantaneous vol |
|---|---|---|---|---|
| FAST-HEAVY | 0.035 | 0.005 | 0.04 | 20% |
| SLOW-HEAVY | 0.005 | 0.035 | 0.04 | 20% |

---

# PART 5 — DATA RANGES

## 5.1 PINN training domain (`nifty_multifactor_v4/config.json`)

| variable | range | sampling |
|---|---|---|
| $x=\log(F/K)$ | $[-0.36,\,+0.36]$ → $S/K\in[0.698,\,1.433]$ | uniform |
| $\tau$ | 7 days to 2 years (0.0192 – 2.0 y) | uniform in $\log\tau$ |
| $v_s$ (slow variance) | $[1.5\times10^{-4},\,0.18]$ → 1.2% – 42% vol | uniform in $\log v_s$ |
| $v_f$ (fast variance) | $[1.0\times10^{-3},\,0.22]$ → 3.2% – 47% vol | uniform in $\log v_f$ |
| level scale $s$ | $[0.7,\,3.2]$ | uniform |
| sampler | 5-dimensional Latin hypercube, one seed per split | |

**Outside this box the network is untrained**, and every figure in the project either stops at the
boundary or labels the extrapolation.

Dataset sizes: teacher train 100,000 (seed 93101); collocation 18,000 (93103); v5 development 8,192
(93201); v5 **untouched final** 16,384 (93202); PDE final 4,096 (93203); short-maturity ATM
diagnostic 6,144 (93204, $\tau\le30$ d, $|x|\le0.12$); RAD pool 120,000 (93205).

## 5.2 Calibration bounds (market tests, `btc_multifactor_v1/config.json`)

| parameter | bound |
|---|---|
| $\kappa$ (single) | $[0.05,\ 50]$ |
| $\kappa_s$ | $[0.05,\ 20]$ |
| $\kappa_f - \kappa_s$ (gap) | $[0.05,\ 100]$ |
| $\theta_i$ | $[0.001,\ 4]$ |
| $v_{i,0}$ | $[0.001,\ 4]$ |
| $\sigma_i$ | $[0.01,\ 10]$ |
| $\rho_i$ | $[-0.99,\ 0.99]$ |
| $\eta_i$ (strict-Feller variant) | $[0.01,\ 0.999]$ |
| $\sigma_{BS}$ | $[0.05,\ 5]$ |

Optimiser: differential evolution (popsize ×6, 25 iterations, seed 20260919) followed by 12
bounded least-squares starts, max 300 function evaluations each, all starts retained.

**Objective** (identical for every model):

$$\min \;\Big\langle \Big(\frac{c_{model}-c_{market}}{\max(\nu,\,10^{-4})}\Big)^2\Big\rangle$$

Dividing by vega makes this approximately an implied-volatility residual.

## 5.3 Market data filters

**Crypto (Deribit)** — `btc/eth/sol/xrp/hype_multifactor_v1`:

| filter | value |
|---|---|
| maturity | 3 – 400 days |
| moneyness | $\lvert x\rvert \le 1.0$ |
| implied vol | 0.10 – 3.00 |
| option type | out-of-the-money only |
| minimum price | 0.0005 coin (BTC/ETH), 2 ticks (USDC-quoted) |
| quotes per expiry | ≥ 3 |
| expiries per date | ≥ 5 (BTC/ETH), ≥ 4 (thin markets) |
| window | 06:00–08:00 UTC (BTC/ETH), full day (thin markets) |

**Equity / index (CBOE)** — `hardcoded_v1`, `dh_relevance_selection_v1`:

| filter | value |
|---|---|
| maturity | 7 – 730 days |
| moneyness | $\lvert x\rvert\le0.36$ (the PINN domain) |
| quote | mid of bid/ask, with bid > 0, ask > bid, ask ≤ 3·bid |
| option type | out-of-the-money only |
| implied vol | 0.02 – 3.00 |
| quotes per expiry | ≥ 6 |
| forward / discount | from the put-call parity regression $C-P = D\,F - D\,K$, requiring $R^2\ge0.999$ |

## 5.4 Range of the quantities plotted

| quantity | range used |
|---|---|
| $S$ (controlled figures) | $0.70K$ to $1.30K$ with $K=100$; ATM zoom $0.85K$–$1.15K$ |
| $\tau$ (curves) | 7 – 730 days, 300–500 grid points |
| $\tau$ (panels) | 30, 90, 180, 365, 730 days (predeclared) |
| calendar time $t$ | 0 to $T$, with $\tau = T-t$; PINN curves stop at $\tau=7$ d |
| $c = C/F$ | $[0,1)$ by construction |
| $C$ (controlled) | 0 to ≈ 30 with $K=100$ |
| time value $C-\max(S-K,0)$ | 0 to ≈ 4 at 90 days |
| implied volatility | ≈ 15% – 26% (controlled), 9% – 50% (market) |

---

# PART 6 — BLOCK ARCHITECTURES

See `figures/pinn_architecture_comparison.png` for the side-by-side diagram.

**Locked v4 (269,825 parameters)**

```
(x, v_s, v_f, τ) ─┐
                  ├─► 24 engineered features ─► Linear(24,256) ─► tanh
(κ,θ,σ,ρ)×2 ──────┘                             └─ [Linear(256,256) ─► tanh] ×4
                                                          │
                                                          ▼
                                                   Linear(256,1) = head
                                                          │
                              analytic baseline √v̄  ×  exp(1.8·tanh(head))
                                                          │
                                                          ▼
                                              Black formula  ─►  C/K
```

**Improved v5 (282,632 parameters, +4.7%)** — same inputs, same features, same baseline, same bounded
correction, same Black layer. Only the body and the training mechanics change:

```
24 features ─┬─► U = tanh(a·W₁f+b₁) ─┐
             ├─► V = tanh(a·W₂f+b₂) ─┤   gated update, 5 levels:
             └─► h = tanh(a·W f+b)  ─┘   Z = tanh(a_ℓ·W_ℓ h + b_ℓ)
                                          h ← (1−Z)⊙U + Z⊙V
```
plus **layer-wise adaptive tanh** ($a_\ell$ learned), **gradient-norm loss balancing** and
**residual-adaptive collocation**.

**Measured effect** on the untouched 16,384-point test set:

| | locked v4 | improved v5 |
|---|---|---|
| price RMSE vs exact DH | 1.062e-5 | **6.610e-6** |
| IV RMSE (vol points) | 0.1175 | **0.0738** |
| worst-case price error | 1.523e-4 | **5.938e-5** |
| PDE residual | **0.0309** | 0.0558 (worse) |

---

## Where each equation lives in the code

| equation | file |
|---|---|
| Black-76 price, vega, implied vol | `experiments/btc_multifactor_v1/engine.py` |
| implied-vol inversion and validity | `src/mentor_dh_pinn/regular_pinn_data.py` |
| factor characteristic exponent | `src/double_heston_reference.py::_factor_exponent` |
| Fourier pricing integral | `experiments/btc_multifactor_v1/engine.py::Grid` |
| admissibility, Feller, disk | `engine.py::admissible`, `src/constraints.py` |
| baseline, features, correction, PDE residual | `src/mentor_dh_pinn/regular_pinn_torch.py` |
| loss, gradient balancing, RAD | `experiments/dh_pinn_v5/train.py` |
| architecture variants | `experiments/dh_pinn_v5/v5_models.py` |
| published parameters, domains | `experiments/nifty_multifactor_v4/config.json` |
| calibration bounds and objective | `experiments/btc_multifactor_v1/config.json` |
