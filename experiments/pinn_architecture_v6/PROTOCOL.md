# Predeclared architecture/training ablation — 2026-09-20

Baseline reproduction and 200-step diagnostic continuation completed before candidate implementation.
See root CURRENT_PINN_BASELINE.md and artifacts/baseline_reproduction.csv.
Current best is v5 SHARED, not C1. It already passes all four original gates.

Scope: improve numerical fidelity only; no market quote reads, calibration, parameter-domain changes,
teacher changes, scenario edits or gate relaxation. Previous tests are already exposed. A new synthetic
fidelity set (4096 points seed 206104; 512 PDE points seed 206105) is generated only AFTER selection and
all checkpoints/code are hashed. Final test is evaluated once, selected candidate and frozen baseline
only, using common points. No reselection on those results. No final market test is opened.

The inherited model uses exact terminal payoff and call bounds. A4 / ARCH_D_HARD_TERMINAL is a mathematical
no-op, not a newly trained candidate: payoff error is already exactly zero. An additive payoff + tau*N
ansatz introduces the spatial payoff kink at positive tau; no valid reason to replace the current smooth
positive-tau variance ansatz was found. Finite x-domain edges are not S=0 or infinity; exact teacher edge
errors will be measured, not forced to an incorrect asymptotic boundary value.

Observed diagnosis: median IV/teacher scaled gradient ratio .0355 (minimum .00713); PDE/teacher median
.3502; short-time PDE RMSE .0563 versus .0161 in medium times. Largest price errors occur at long times.
These diagnostics justify gradient-balancing and RAD, not a claimed proof of spectral bias. Fourier
features and gPINN are not included. Domain keeps its fixed kappa/rho family, with kappa ratio ~11.3.

To respect 'do not start from scratch', every new branch starts as a zero correction to the reproduced
v5 checkpoint. The full inherited checkpoint is frozen in new-branch arms. A faithful published modified
MLP is used INSIDE a residual correction branch; this is explicitly not replacement/retraining of the
entire original MLP. U=tanh(Wu f+bu), V=tanh(Wv f+bv), H=tanh(Wh H+b), H=H*U+(1-H)*V at EVERY hidden
layer, then linear head. This matches the official GradientPathologiesPINNs forward_pass M3/M4 and JaxPI
ModifiedMlp; gates are tanh, not sigmoid. Zero output head preserves the inherited function. Output residual
.05*tanh(head) changes log IV only; old pricing/PDE code is reused unchanged.

Ordered arms, seeds 17 and 43 in every arm (two independent parent checkpoints; no fabricated third seed):
1. A0_CONTINUE: existing v5 shared branch weights alone, equal additional optimizer steps.
2. CONTROL_PLAIN: new width-104, five-hidden-layer plain tanh correction.
3. ARCH_A_MODIFIED_MLP: new width-96, five-hidden-layer published modified MLP correction.
4. ARCH_B_MODIFIED_MLP_GRADBAL: A plus adaptive loss balancing.
5. ARCH_C_MODIFIED_MLP_GRADBAL_RAD: B plus RAD.
Arms start independently from the same parent checkpoint, not the previous arm's trained weights.
CONTROL_PLAIN isolates new branch/capacity/conditioning from the modified-MLP mechanism.
All new branch totals must be <=1.25 times the 275684 per-seed baseline parameters.

Input conditioning: new branches use the same 26 engineered inputs already present in v5, with train-only
minimum/maximum affine scaling to [-1,1]. Constant features use scale 1. No clipping or domain changes.
CONTROL_PLAIN and A/B/C share the transformation. The parent representation stays unchanged.

Common budget: 2000 Adam steps at 2e-4 cosine-decayed to 1e-5, 512 teacher labels and 64 collocation points
per step, then 80 strong-Wolfe L-BFGS iterations, history 20, first 4096 training labels and 256 fixed
collocation indices. No weight decay. Current v4 loss normalizers unchanged. Save before/after L-BFGS
metrics. Log raw/scaled component losses, parameter gradients, effective weights, time, process peak RSS.
Both training-step budgets and wall times are disclosed; adaptive diagnostics are additional compute.

Balancing: official JaxPI grad_norm rule, mean L2 gradient norm / each component L2 gradient norm, followed
by EMA with beta=.9. Compute on normalized price, IV and PDE terms every 100 Adam steps; zero-norm
components keep their old weight. Convexity remains a separate .1-weight constraint term; hard terminal
and absent boundary penalties must not cause division by zero. This is the later official grad_norm
variant, not the original paper's max/mean statistic. Freeze adaptive weights during L-BFGS to maintain a
fixed closure objective. Stop and record failures for nonfinite losses/gradients or weights outside
[1e-6,1e6]; do not silently clip weights or gradients.

RAD: existing 18000-point initial collocation set. At steps 500,1000,1500 generate a fresh 8192-point LHS
candidate pool using the unchanged sampler/domain, score |scaled PDE residual|, and use probabilities
proportional to |R|/mean|R| +1 (Wu et al. RAD k=1,c=1). Replace 4096 of 18000 collocation locations,
keeping 13904 original global points. Also guarantee at least one collocation point per marginal bin
in each minibatch: short/medium/long maturity; OTM/ATM/ITM; low/high each variance; fast/slow-heavy.
The remaining slots use the current distribution. The same stratified minibatch rule is used in ALL arms;
A0 continuation therefore differs in sampling from historical v5, which is explicitly a control.
Candidate scoring and replacement costs are logged, density plots saved, and active count stays 18000.

Selection: only post-LBFGS development ensemble metrics. Eligible new model passes RMSE<=2e-5,
P95<=5e-5, max<=2e-4, IV RMSE<=.002 decimal, and improves baseline development price RMSE. Additional
honesty diagnostics report regional regressions, curve violations and compute even if primary gates pass.
Rank eligible arms by price RMSE, then IV RMSE, total parameter count, inference latency. Select NONE if
none eligible; do not promote a nonpassing candidate. The frozen baseline remains available.

Conditional E: only if A/B/C all fail to pass/improve, test ARCH_E_ADAPTIVE_ACTIVATION against A alone
with identical settings and per-hidden-layer tanh(a_l x), scales initialized at 1, no neuronwise scaling
or extra slope penalty. The original activation paper motivates it; no assumed benefit. No other retry.
If any A/B/C improves and passes, E is not required. D/F/G decisions are recorded as not trained, not
invented ablation rows. Two seeds imply limited statistical robustness; report mean/std/best/worst.

Curves: fixed published base parameters; x grid [-.36,.36] 81 points; 41 geometric maturities 7–730d.
Same teacher and axes for all candidates. Evaluate exact teacher at finite-domain edges. Measure true
spot monotonicity and convexity through price derivatives, holding strike/carry/tau/states fixed.
Positive-tau curves and exact payoff at zero are separate; never differentiate payoff kink at tau=0.

Primary references (official code snapshots in research/):
- Wang, Teng, Perdikaris 2021, https://arxiv.org/abs/2001.04536 and official GradientPathologiesPINNs.
- Wang, Yu, Perdikaris 2022, https://arxiv.org/abs/2007.14527 (motivation only; no claim of an NTK eigensolver).
- Wang et al. 2023, https://arxiv.org/abs/2308.08468 and official JaxPI grad_norm implementation.
- Wu et al. 2023, https://arxiv.org/abs/2207.10289 and lu-group/pinn-sampling RAD implementation.
- Sukumar, Srivastava, https://arxiv.org/abs/2104.08426 (hard-condition assessment; no generic distance function).
