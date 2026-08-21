# Archive-2 PDE Loss — Term-by-Term Audit (Phase D)

Source: `src/dheston/models/losses.py:78-134` (`pde_residual_loss`), inputs
from `price_double_heston_torch` (`src/dheston/pricing/heston.py`).
Canonical PDE (forward tau): see `derivations/CANONICAL_DOUBLE_HESTON_PDE.md`.

Legend: CORRECT / LIKELY CORRECT / INCORRECT / INCOMPLETE / UNVERIFIED.

| Implementation (line) | Claimed mathematical term | Verdict | Notes |
|---|---|---|---|
| `prices = price_double_heston_torch(...)` (104) | option value V(S, v1, v2, tau; params) | INCORRECT OBJECT | V is the analytic COS pricer, not a learned solution; residual self-referential, ~0 for any params if derivatives were correct (FINDINGS F3) |
| `spot = ...requires_grad_(True)` (97) | spot as differentiable state | CORRECT | genuine leaf created before forward |
| `tau = ...requires_grad_(True)` (99) | time-to-maturity as differentiable state | CORRECT | genuine leaf; tau-convention correct (U_tau form) |
| `strike` (98) | fixed strike | CORRECT | V_S at fixed K is the right convention |
| `d_tau = _safe_grad(prices, tau)` (106) | V_tau | CORRECT FORM | nonzero in instrumentation; includes spurious d(truncation range)/d(tau) through COS cumulants |
| `delta = _safe_grad(prices, spot)` (107) | V_S | CORRECT FORM | nonzero (0.615/0.612 in instrumentation) |
| `gamma = _safe_grad(delta, spot)` (108) | V_SS | CORRECT FORM | second-order via create_graph; works |
| `v01 = chosen_params[:, 0]` (110) | v_fast state variable | INCORRECT | fresh post-hoc view: NOT in autograd graph -> all v-derivatives unused |
| `v02 = chosen_params[:, 5]` (115) | v_slow state variable | INCORRECT | same defect |
| `d_v01 = _safe_grad(prices, v01)` (121) | V_v1 | INCORRECT | exactly 0.0; true value 28.49 / 13.15 (instrumented) |
| `d_v02 = _safe_grad(prices, v02)` (122) | V_v2 | INCORRECT | exactly 0.0 |
| `cross_sv01 = _safe_grad(delta, v01)` (123) | V_Sv1 | INCORRECT | exactly 0.0 |
| `cross_sv02 = _safe_grad(delta, v02)` (124) | V_Sv2 | INCORRECT | exactly 0.0 |
| `d2_v01 = _safe_grad(d_v01, v01)` (125) | V_v1v1 | INCORRECT | grad of a zero constant -> 0.0 |
| `d2_v02 = _safe_grad(d_v02, v02)` (126) | V_v2v2 | INCORRECT | 0.0 |
| `diffusion = 0.5*(v01+v02)*spot^2*gamma` (128) | 1/2 (v1+v2) S^2 V_SS | CORRECT FORMULA | coefficient/structure match canonical PDE |
| `drift = (r-q)*spot*delta - r*prices` (129) | (r-q) S V_S - r V | CORRECT FORMULA | sign/discount convention correct |
| `factor_one = k1(th1-v1)V_v1 + rho1 sigma1 v1 S V_Sv1 + 1/2 sigma1^2 v1 V_v1v1` (130) | slow/fast factor operator | CORRECT FORMULA | cross coefficient rho*sigma*v*S matches derivation; NO V_v1v2 term — correct for independent variance BMs |
| `factor_two = ...` (131) | second factor operator | CORRECT FORMULA | same |
| `residual = d_tau - (diffusion+drift+f1+f2)` (132) | U_tau - (L U - r U) | CORRECT FORMULA | forward-tau sign convention correct |
| `scale = prices.abs().clamp_min(1)` (133) | relative residual scaling | LIKELY CORRECT | detached; relative scaling defensible |
| `mean((residual/scale)^2)` (134) | loss aggregation | LIKELY CORRECT | — |
| point subsampling `indices[::step][:max_points]` (86-91) | collocation subset | LIKELY CORRECT | deterministic stride; fine |
| `boundary_penalty` (48-59) | no-arbitrage band | NOT BOUNDARY PHYSICS | static band the analytic pricer satisfies identically; no terminal/edge PDE conditions exist in this stack |
| no terminal-condition loss | U(S,v,0) = payoff | INCOMPLETE | absent entirely |
| `_safe_grad` allow_unused -> zeros (62-75) | robustness shim | INCORRECT DESIGN | converts autograd non-connectivity into silent zeros — the direct enabler of the F2 defect |

Summary: **the assembled formula is the correct canonical Double Heston PDE,
but the autograd inputs for all five variance-state derivatives are not in the
computation graph, so the evaluated operator is missing the entire variance
dynamics** (instrumented: residual == residual-without-v-terms exactly). A
correct construction must make the state entries genuine leaves assembled into
the parameter vector (as demonstrated for the canonical pricer in
`probe_residuals.py: canonical_pde_terms`, rel. residual <= 1.3e-15) — and,
per F3, should target a learned pricing function rather than the analytic
pricer.
