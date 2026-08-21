# Adversarial Review Report — Node C PDE/Physics Audit

Reviewer: independent adversarial agent (2026-08-22 ~01:55 IST), re-derived
the mathematics and re-executed the decisive numerical experiments from a
clean checkout (repo untouched). Verdicts below are the reviewer's own; Node
C applied the required corrections (see FINDINGS F13).

---

## Item verdicts

(a) 4-BM two-spot-driver construction — **CONFIRMED**. The CF (b = kappa -
rho*sigma*iu at src/double_heston.py:122, per-factor exponents summed at
193-197, drift at 194) is exactly the product of two single-factor Heston
CFs; factorization forces d<W_s,W_f>=0 and d<B_s,B_f>=0. A single-driver
model yields non-affine covariation rho_i*sigma_i*S*sqrt(v_i(v_s+v_f)) and
cannot produce this CF in closed affine form. Wording correction: the "if and
only if" in derivation §1.2 is an equality-in-law statement. [APPLIED]

(b) PDE correctness and tau-convention — **CONFIRMED**. Generator re-derived:
(dS)^2 = S^2(v_s+v_f)dt, d<S,v_i> = rho_i*sigma_i*S*v_i*dt, d<v_s,v_f> = 0;
exact PDE match including cross coefficients, no V_vsvf, 1/2x2 bookkeeping,
tau convention. Numerical certification values match probe_results.json
exactly. The separate "independent re-derivation" was UNVERIFIABLE from
committed artifacts — now committed (INDEPENDENT_CROSS_DERIVATION.md), and
the reviewer's own re-derivation substitutes for it. [RESOLVED]

(c) Autograd through a Fourier pricer by time-homogeneity — **CONFIRMED**,
with domain caveats: identity holds for the exact price; the discrete pricer
satisfies the PDE only up to quadrature error (1e-15 at the 8 moderate
points; grows at ultra-short maturities per F10; broadened sweep <= 1.6e-8).
COS truncation range is state-dependent (measured negligible ~1e-12 for
delta). The leaf-wired construction is correct.

(d) Post-hoc views cannot receive gradients — **CONFIRMED definitively
("airtight")**. Reviewer reproduced: _safe_grad(prices, v01_posthoc) = 0.0;
even a view created BEFORE the forward pass receives 0.0 if never consumed;
gradient w.r.t. the actual ancestor = 28.4851 (matches Node C's 28.49).
Mechanism: the pricer consumes the whole 2-D chosen_params and creates its
own SelectBackward nodes (heston.py:156-157); losses.py's views are children,
not ancestors. Without allow_unused, torch.autograd.grad raises (verified);
with it, None -> zeros (losses.py:73-74). d2_v01 is zeros-of-zeros via the
requires_grad guard. NO mechanism can make these gradients non-zero.

(e) Disk sufficient-not-necessary — **CONFIRMED**. det a verified symbolically
and numerically (eigendecomposition positive for rho = (-0.8,-0.7), (0.9,0.9),
(-0.95,-0.95), all with disk > 1). Sharpening: even uniform PD of the
observable (dS/S, dB_s, dB_f) correlation matrix does not require the disk
(det proportional to (1-rho_s^2)w_s + (1-rho_f^2)w_f > 0 iff |rho_i| < 1).
[APPLIED to §1.3]

(f) Additivity identity — **CONFIRMED** (exact algebra: b, d, g independent
of (theta, v0); E = theta*A + v0*B). Reproduced 1.776e-15.

(g) Two-half-factors == single Heston — **CONFIRMED** (immediate from (f);
COS standard-Heston parameter order (v0, kappa, theta, sigma, rho) checked —
cross-check is apples-to-apples). Domain caveat: stricter half-factor Feller
gate kappa*theta > sigma^2. 3.24e-11 reproduced.

(h) boundary_penalty band — **CONFIRMED** as the CORRECT tight European
no-arbitrage bounds under continuous dividends (call lower bound
max(Se^{-q tau} - Ke^{-r tau}, 0) — exactly what is implemented; the variable
names intrinsic_call/put are misnomers). Phrasing fix: "satisfied up to
quadrature error" (smoke artifact train_boundary = 1.12e-11; COS clamps
prices at 1e-8), not "identically". Classification "not boundary physics"
correct. [APPLIED]

(i) Parameter-contract mapping — **CONFIRMED**. Factor 1 = FAST proven by
three independent code facts (transforms.py:57 sampling, 89-90 sandwich,
losses.py:44-45 penalty). Index table and adapter permutation correct.
Semantic-conflict demonstration reproduced (gaps -2.248/-2.244, disk 1.805).

(j) FINDINGS F1-F12 — substantially accurate. Corrections applied:
F1 certification domain; derivation §3 stale O(sigma) -> O(sigma^2) and
terminal phrasing; F5 gap-value transcription (-2.248 slow / -2.244 fast);
F9/F12 dead-field attributions (u_max/alpha/integration_eps are never
consumed; saturation comes from float64-resolution CF decay, not a cap);
"consistent with train_pde 8.9" footnoted as qualitative. F4/F6/F7 verified
including artifact values and the research-control prohibition verbatim
(docs/RESEARCH_CONTROL_AND_CURRENT_STATUS.md:36).

## Canonical-stack autograd scan — CLEAN

Repo-wide: torch.autograd.grad / allow_unused appear ONLY in Archive-2's
_losses (and Node C tests). train_pinn.py / pinn_model.py /
torch_double_heston.py have no such pattern; .detach() uses are validation
guards. The defect class is unique to _safe_grad.

## New minor findings (reviewer)

1. Dead FourierConfig fields (alpha, u_max, integration_eps) — latent
   operational trap; root cause of the F9/F12 misattributions.
2. float32 FD artifact in Node C's delta_contamination probe (torch.tensor
   default dtype): FD appeared off by 3.4e-4; test-harness artifact, fixed to
   float64 (now ~1e-9); F3's autograd-vs-GL comparison unaffected.
3. Latent non-differentiable early-return at maturity == 0.0 in
   torch_double_heston.py:295-296 (currently unreachable via validated entry
   points).
4. Dead compute in boundary_penalty (_expand_batch_to_points result deleted,
   losses.py:49-50).
5. Naming: intrinsic_call/put hold discounted European bounds.

## Bottom line (reviewer)

All load-bearing claims — (a), (b), (d), (e), (f), (g), (i), and the
F2/F3/F4/F5/F6/F7 classifications — survive adversarial attack, including
independent re-execution. Nothing needs retraction. Required corrections
(listed above) have been applied by Node C in commit "apply
adversarial-review corrections". New findings are minor evidence-hygiene
items; none affect conclusions.
