# Current reproduced PINN baseline — 2026-09-20

Read-only recovery and numerical reproduction completed before any v6 architecture change.
Current best selected existing model: v5 SHARED, inherited from v4 C3, seeds 17 and 43.
No later experiment directory was found. v3, v4 and v5 source hashes all verify.

| Property | Recovered value |
| --- | --- |
| Core | Regular variance PINN, 5 hidden layers, width 256, tanh |
| Extension | 3 shared width-32 two-hidden-layer tanh branches, equal weights, bounded additive log-IV correction |
| Parameters | 275,684 per seed; 551,368 for two-seed price ensemble |
| Original C3 parameters | 269,825 per seed |
| Current extension trainable parameters | 5,859 per seed; C3 weights frozen during v5 training |
| Independent coordinates | log(F/K), v_slow, v_fast, tau |
| Conditional inputs | kappa, theta, sigma, rho for each factor; 24 engineered features plus 2 exact decay features in v5 branches |
| Precision | float64, CPU; one PyTorch thread per training process |
| C3 optimization | 40,000 Adam steps, cosine schedule 1e-3 to 1e-5, then 300 L-BFGS iterations with strong-Wolfe line search |
| v5 optimization | 800 Adam steps, 2e-4 to 1e-5, then 60 L-BFGS iterations |
| Teacher | Existing exact Double Heston, 100,000 training labels; 4,096 development points; all price labels retained |
| Collocation | 18,000 deterministic LHS points; uniform x and scale, logarithmic tau and both states; random minibatches |
| Loss | price MSE/(2e-5)^2 + valid IV MSE/.002^2 + .1 scaled-PDE MSE/.01^2 + .1 negative-convexity MSE |
| Weights | Static; no gradient balancing or adaptive collocation in selected v5 |
| Hard terminal | Exactly max(exp(x)-1,0) in strike-discount-normalized units at tau=0 |
| Output | Positive implied variance: w=tau*expected_average_variance*exp(2*correction); normalized analytic Black call evaluated at learned w |
| Labels are not Black–Scholes | Black is an output transform; teacher labels remain exact DH |
| Rate/spot/strike conditioning | Homogeneous forward normalization removes separate S,K,r,q; x=log(F/K), NOT log(S/K) when carry is nonzero |

Reproduced development ensemble metrics:

| Price RMSE | IV RMSE (vol points) | P95 | Max | Mean per-seed scaled PDE RMSE |
| --- | --- | --- | --- | --- |
| 1.0705229530e-5 | 0.1159069695 | 2.2330423592e-5 | 1.2333108985e-4 | 0.0210189970 |

PDE values above use the original v5 256-point diagnostic subset, not the full domain.
The previously exposed v5 synthetic test price RMSE was 1.05478397e-5; it cannot be relabeled untouched.

Current machine timing: median two-seed batch-1024 inference 23.51 ms (43,561 quotes/s), 20 repetitions after warm-up.
Timing is environment-dependent, not inherited training hardware equivalence.
Historical C3 training: 2711.63 / 2681.99 seconds; v5 incremental training: 44.34 / 47.24 seconds.
Historical logs contain no gradient-norm trajectories or measured peak memory; these cannot be reconstructed retroactively.
V6 will record a separately labelled continuation diagnostic rather than fabricate historical gradients.

Current checkpoints:
- experiments/nifty_multiscale_v5/artifacts/SHARED_s17/weights.pt
- experiments/nifty_multiscale_v5/artifacts/SHARED_s43/weights.pt

C1/C2/C3 development price RMSE: 3.3248e-5 / 1.8042e-5 / 1.0708e-5.
C1 failed several gates; C2 failed maximum error; C3 passed and was selected. V5 SHARED narrowly improved C3.
Frozen gates retained: price RMSE <=2e-5, P95 <=5e-5, maximum <=2e-4, IV RMSE <=.002 decimal (0.2 vol points).

Input audit: most engineered features are O(1), but the inherited log-state transforms use generic .01–.3 ranges.
On development data the slow-state feature spans [-3.469,.699], fast-state [-2.354,.817].
Some kappa/rho/Feller-ratio features are constant in this fixed structural family. This is NOT a ten-parameter-varying domain.
No change to that domain is authorized. Any new branch may condition these same features with train-only affine statistics;
comparison controls must share that conditioning. The inherited network representation remains unchanged.

Test status: v4 final market metadata records opening on 2026-09-17. Its results and v5 market results are already exposed.
V3 was not locked. V6 will read no market quote data, create no market forecasts, and reserve a new synthetic fidelity seed
because both prior synthetic fidelity splits have been opened. Selection uses the same inherited development set only.

Source/plotting audit: v4 uses price/IV/bucket curves and controlled-analysis surfaces; v5 adds shared residual and maturity
ablations. Shared price, feature, PDE, sampling and exact-pricing implementations will be reused; old files will not be changed.
