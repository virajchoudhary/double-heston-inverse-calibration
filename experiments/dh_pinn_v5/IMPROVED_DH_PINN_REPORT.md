# Improved Double Heston PINN: architecture study and market-residual hybrid

2026-09-25. Everything below was specified in `config.json` and frozen before any v5 training. Figures are in `figures/`, artifacts in `artifacts/`.

**Headline.** The improved network is a **1.6× more accurate surrogate** for exact Double Heston (price RMSE 1.06e-5 → 6.61e-6 on an untouched test set), and that improvement moves real SPX pricing error by **0.003 vol points out of 5.8**. A small market-residual head cuts held-out SPX error by roughly two thirds, but under an identical head the **Double Heston prior does not beat the Single Heston prior**.

---

## 1. The existing architecture

`src/mentor_dh_pinn/regular_pinn_torch.py`, locked as candidate C3 in `nifty_multifactor_v4`.

- **12 inputs:** coordinates x = log(F/K), v_slow, v_fast, τ; and, per variance factor, κ, θ, σ, ρ.
- **24 engineered features, nothing learned:**
  - 4 general: x/0.36; z = x/√(τ·v̄ + x²/64); tanh(z/1.5); log τ scaled to [7 days, 2 years].
  - 9 per factor: log vᵢ, log κᵢ, log θᵢ, ρᵢ, tanh(κᵢτ), νᵢ = tanh(σᵢ√τ/√v̄), ρᵢνᵢ, tanh(½ log(vᵢ/θᵢ)), and a Feller ratio 2log(1+ηᵢ)/log 4 with ηᵢ = σᵢ/√(2κᵢθᵢ).
  - 2 cross-factor: the slow factor's variance share, and tanh(z/1.5)·Σρᵢνᵢ.
- **Body:** Linear(24,256) → tanh, then four Linear(256,256) → tanh. No skip connections. **269,825 parameters, float64.**
- **Output:** a bounded correction 1.8·tanh(head) multiplying an analytic baseline volatility √v̄, where v̄ = Σᵢ[θᵢ + (vᵢ−θᵢ)(1−e^(−κᵢτ))/(κᵢτ)]; the implied volatility then goes through Black. The payoff at expiry is exact by construction and price bounds cannot break.
- **Training:** 100,000 exact Fourier prices, 18,000 collocation points, price/IV/PDE/convexity loss, Adam 40,000 steps then 300 L-BFGS steps, two seeds averaged.

The prior-plus-correction design is preserved throughout v5; only the body and the training mechanics change.

## 2. Existing limitations, quantified

| claim | measurement |
|---|---|
| fidelity ceiling near 1e-5 | untouched-set price RMSE **1.062e-5**, IV RMSE **0.1175** vol points |
| worst case is 14× the RMSE | price max error **1.523e-4** |
| short maturity is the weak region for the network | 7–30 day IV RMSE **0.223** vs 0.005 beyond a year |
| plain tanh stack trains slowly | at an identical 12,000-step budget the plain body reaches only **6.14e-5** |
| the ≤30-day **market** failure is NOT a network problem | on the short-ATM diagnostic the locked net is **0.062** vol points, better than its own average; the market error there is 7.5 |

That last row is the key diagnostic: the short-maturity market gap cannot be repaired by a better solver.

## 3. Architecture modifications, and the ablation

Identical budget for every stage: seed 17, 12,000 Adam steps, 512 label and 128 physics points per step, 300 L-BFGS steps, same data.

| stage | change | params | train s | dev price RMSE | dev IV RMSE | PDE |
|---|---|---:|---:|---:|---:|---:|
| locked v4 | 40,000 steps, 2 seeds | 269,825 | 5,394 | 1.081e-5 | 0.1125 | 0.031 |
| A0 | plain tanh stack | 269,830 | 1,491 | 6.144e-5 | 0.3513 | 0.138 |
| A1 | + residual blocks, h ← h + αₗ·tanh(Wh+b) | 269,834 | 1,716 | 1.985e-5 | 0.1904 | 0.044 |
| A2 | + layer-wise adaptive tanh(aₗz) | 269,834 | 1,801 | 1.572e-5 | 0.1575 | 0.038 |
| A3 | + gradient-norm loss balancing | 269,834 | 1,542 | 2.549e-5 | 0.1157 | 0.082 |
| A4 | + residual-adaptive collocation (RAD) | 269,834 | 1,333 | 2.475e-5 | 0.1197 | 0.036 |
| **A5** | **gated block instead of residual** | **282,632** | **1,884** | **1.471e-5** | **0.1051** | **0.020** |

- **Residual connections are the single biggest win:** 3.1× lower price error than the plain stack at the same budget and parameter count.
- **Adaptive tanh adds 21%.** Learned scales settle at 1.1–1.4 (residual stages) and 0.6–1.0 (gated).
- **Gradient balancing trades price for IV and PDE:** IV 0.158 → 0.116, price error rises. It reweights away from the price term by design.
- **RAD mainly repairs the physics residual** (0.082 → 0.036) at no extra cost and slightly less time.
- **The gated block (Wang, Teng & Perdikaris) wins on every metric** for 4.7% more parameters.

**Winner by the frozen rule** (lowest development price RMSE; a simpler stage within 5% would have won): **A5**. It was then retrained at the full locked recipe, two seeds, and only then was the final set opened.

## 4. Exact-DH fidelity: old vs improved (untouched 16,384-point set, opened once)

| metric | locked v4 | improved v5 | ratio |
|---|---:|---:|---|
| price RMSE | 1.062e-5 | **6.610e-6** | 1.61× better |
| price P95 | 2.258e-5 | **1.419e-5** | 1.59× |
| price max | 1.523e-4 | **5.938e-5** | 2.56× |
| IV RMSE (vol points) | 0.1175 | **0.0738** | 1.59× |
| 7–30 day IV RMSE | 0.2232 | **0.1429** | 1.56× |
| short-ATM diagnostic IV RMSE | 0.0618 | **0.0317** | 1.95× |
| scaled PDE residual | **0.0309** | 0.0558 | 1.8× **worse** |
| parameters | 269,825 | 282,632 | +4.7% |
| training seconds (2 seeds) | 5,394 | 14,205 | 2.6× |
| inference latency | 14.2 µs/quote | 24.8 µs/quote | 1.7× slower |

**Against the predeclared targets:** price RMSE < 7e-6 **met** (6.61e-6); stretch < 5e-6 not met. IV RMSE < 0.07 **just missed** (0.0738); stretch < 0.05 not met.

**Honest costs:** the PDE residual got worse, because gradient balancing moved weight toward the supervised terms; and inference is 1.7× slower per quote. The accuracy gain is real but not free.

## 5. Real SPX benchmark (10,706 quotes, VIX 14.2)

| model | nothing fitted | one level scale fitted |
|---|---:|---:|
| Black-Scholes, fixed vol | 6.855 | 7.271 |
| Single Heston, published | 5.359 | 5.788 |
| Double Heston, published (exact) | **5.344** | 5.818 |
| DH-PINN locked v4 | 5.365 | 5.850 |
| DH-PINN improved v5 | 5.364 | 5.847 |

By maturity, level scale fitted:

| maturity | BS | SH | DH exact | DH-PINN v4 | DH-PINN v5 |
|---|---:|---:|---:|---:|---:|
| ≤ 30 days | **7.33** | 7.47 | 7.49 | 7.51 | 7.55 |
| 30–90 days | 7.75 | 5.57 | 5.51 | 5.55 | **5.50** |
| 90–365 days | 7.00 | 4.30 | 4.45 | 4.48 | 4.45 |
| > 365 days | 6.18 | 3.66 | **3.56** | 3.57 | 3.56 |

## 6. Why pure architecture work has a market ceiling

The improved network reduced its distance to exact Double Heston by **0.044 vol points** (0.1175 → 0.0738). Its SPX error changed by **0.003 vol points** (5.850 → 5.847). Meanwhile the market gap is **5.8**.

- network error / market error ≈ **0.0738 / 5.85 ≈ 1.3%**
- Even a *perfect* surrogate (zero network error) would land on exact Double Heston's 5.818. The entire remaining 5.8 is model and parameter misspecification.

The frozen stopping rule (stop once IV RMSE < 0.07) was effectively reached at 0.0738, and the argument does not depend on the last 0.004: **architecture work cannot repair a fixed-parameter model's market error.** So Track B was run.

## 7. Market-residual hybrid (Track B) — clearly a hybrid, not exact Double Heston

Frozen structured split of the SPX surface: checkerboard on (expiry rank + strike rank). **4,271 calibration, 1,082 development, 5,353 held-out.** Held-out quotes are never inputs, never train the head, never select δ_max.

Head, identical for every prior: inputs [x, log τ, prior IV, x/√τ] standardised on calibration statistics; Linear(4,64)→tanh→Linear(64,64)→tanh→Linear(64,1); ΔIV = δ_max·tanh(·) added to the prior IV, then Black. **4,545 parameters**, Adam 4,000 steps, λ = 0.05 on ΔIV², δ_max chosen on development from {0.02, 0.05, 0.10}. Structural parameters never change; only the level scale is fitted, on calibration quotes only.

| prior | held-out IV RMSE, prior only | + identical residual head | mean \|ΔIV\| |
|---|---:|---:|---:|
| Black-Scholes | 7.274 | 2.366 | 4.33 |
| Single Heston | 5.773 | **1.917** | 3.33 |
| Double Heston (exact) | 5.839 | 1.990 | 3.35 |
| Double Heston (improved PINN) | 5.850 | 1.970 | 3.36 |

The hybrid cuts held-out error by about two thirds, from ~5.8 to ~2.0, and beats the predeclared 4.5 and 4.0 targets. **But the gain comes from the residual head, not from the prior.** δ_max chose the largest value on the grid (0.10) for every prior, so the bound binds — the head wants more freedom than the protocol allows.

## 8. Fair prior comparison, the point of the exercise

Same architecture, parameter count, optimiser, budget, calibration data, regularisation and evaluation data for all priors.

| comparison (held-out, 19 maturity×moneyness cells) | mean error difference | cells favouring DH | Wilcoxon p | bootstrap 95% | DH wins? |
|---|---:|---:|---:|---|---|
| Black-Scholes − Double Heston | +0.580 | 13/19 | 0.018 | [+0.087, +1.292] | **yes** |
| Single Heston − Double Heston | +0.020 | 9/19 | 0.340 | [−0.009, +0.079] | **no** |
| improved-PINN prior − exact-DH prior | −0.019 | 8/19 | 0.909 | [−0.063, −0.001] | n/a (the PINN prior is marginally better) |

**Under equal capacity, the Double Heston prior beats Black-Scholes and does not beat Single Heston.** The residual-complexity diagnostic predicted this: after the level fit, the residual left by Double Heston (5.797 vol points RMSE) is indistinguishable from Single Heston's (5.801). It is slightly smoother across maturity (0.678 vs 0.794) but no smaller, so the same small head learns it about equally well.

**No-arbitrage control** (held-out grid): no monotonicity violations for any hybrid. Convexity violations 773–866 of ~5,250 triples and calendar violations ~660 of ~5,330 — but **the raw market quotes on the same grid show 1,087 convexity violations**, so the hybrids are smoother than the data they fit, not broken.

## 9. Maturity and moneyness

Hybrid held-out IV RMSE:

| maturity | BS-hybrid | SH-hybrid | DH-hybrid | DH-PINN-hybrid |
|---|---:|---:|---:|---:|
| ≤ 30 days | 2.69 | **3.06** worse than BS | 3.22 | 3.17 |
| 30–90 days | 2.91 | 1.55 | 1.49 | **1.48** |
| 90–365 days | 1.69 | **0.34** | 0.37 | 0.37 |
| > 365 days | 0.50 | **0.17** | 0.17 | 0.17 |

| moneyness | BS-hybrid | SH-hybrid | DH-hybrid | DH-PINN-hybrid |
|---|---:|---:|---:|---:|
| x < −0.15 | 0.15 | 0.24 | 0.15 | 0.15 |
| ATM | 0.20 | 0.19 | 0.19 | 0.19 |
| x > 0.15 (low strikes) | 4.91 | **3.90** | 4.07 | 4.02 |

- The Double Heston advantage survives only in the **30–90 day** band, where it is genuine but small.
- **The ≤ 30-day weakness remains**, and it is not a network artifact: fixed-parameter Black-Scholes is the best short-dated model both before and after the residual head, because a fixed two-factor shape is wrong there in a way a 4-input correction cannot fix.
- The hardest region for everything is the **deep low-strike wing**, where the SPX smile is far steeper than any published parameter set produces.

## 10. Final architecture

`figures/6_improved_pinn_architecture.png` and `figures/7_hybrid_architecture.png`.

- **Pure PINN (A5):** 24 engineered features → gated block with shared U/V encoders, Z = tanh(aₗ·Linear(h)), h ← (1−Z)⊙U + Z⊙V, five levels, width 256 → bounded correction → analytic baseline → Black. **282,632 parameters.** Trained with gradient-norm loss balancing and residual-adaptive collocation.
- **Hybrid:** that network (or exact Double Heston) as a fixed-parameter prior, plus the **4,545-parameter** bounded residual head.

## 11. Conclusions

1. **Did the improved PINN reproduce exact Double Heston more accurately?** Yes, on an untouched test set opened once.
2. **By how much?** Price RMSE 1.61× better (1.062e-5 → 6.610e-6), worst case 2.56× better, IV RMSE 1.59× better (0.1175 → 0.0738), short-maturity ATM error 1.95× better. Costs: 4.7% more parameters, 1.7× slower inference, and a 1.8× worse PDE residual.
3. **Did that materially improve real SPX pricing?** No. 5.850 → 5.847 vol points, a change of 0.003.
4. **How much was model mismatch rather than network error?** Essentially all of it. Network error is 0.074 vol points against a market error of 5.85, about 1.3%. A perfect surrogate would still sit at 5.818.
5. **Did the small residual extension materially improve held-out SPX accuracy?** Yes, decisively: 5.839 → 1.990 vol points for the Double Heston prior, on quotes never used for anything. This is a hybrid and is labelled `DH_PINN_RESIDUAL`, not exact Double Heston.
6. **Under equal-capacity correction, did Double Heston beat Single Heston and Black-Scholes?** It beat Black-Scholes (p = 0.018, 13/19 cells). It did **not** beat Single Heston (1.990 vs 1.917, p = 0.34, 9/19 cells). The desired ordering DH ≪ SH < BS did not appear, and I have not manufactured it.
7. **Where was the advantage largest?** 30–90 days (1.49 vs 1.55) and the 0.05–0.15 moneyness band (0.81 vs 0.89). Both small.
8. **Did the short-maturity weakness remain?** Yes. Black-Scholes is still best at ≤ 30 days, before and after the residual head. The network is not responsible: its own short-dated surrogate error is 0.03 vol points against a 3–7 vol point market gap.

**What would actually move the market number**, on this evidence: per-date calibration of the structural parameters (as the BTC/ETH experiments in this repository do), or a richer prior for the steep low-strike wing — not more network capacity.

## Operational notes (no protocol change)

- Training gained checkpoint/resume after a machine restart destroyed a run. Same architecture, seeds, budget and schedule.
- The RAD resampling step was rewritten to evaluate the residual in chunks of 2,500 and to draw the weighted sample by the Gumbel top-k method, after the unchunked version exhausted memory on a 16 GB machine. Same candidate pool, same rule, same proportions. The A0–A5 ablation ran under the original implementation; the final training used the chunked one.
- `mean |ΔIV|` values are computed nan-safely in `artifacts/track_b/correction_size.json`; the same field inside `results.json` is NaN because a handful of held-out quotes have no invertible prior implied vol.
