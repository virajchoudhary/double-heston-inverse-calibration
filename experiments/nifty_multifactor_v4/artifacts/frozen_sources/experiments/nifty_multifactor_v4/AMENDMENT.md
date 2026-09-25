# v2 numerical amendment, before any comparison result

v1 manifest `648d82f43feaf1658bf49636a1de915cdc7cdafc8dd21212de9191a6c96d62ba`
is preserved with its failed Phase A record. Seventeen tests passed; one failed
because ordinary adaptive integration reached 500 subdivisions at T=1e-7,
x=-.1. No v1 market selection, baseline fit or neural training occurred.

v2 changes only the short-time independent reference integration: u=z/sqrt(T)
and sine/cosine-weighted quadrature handle the rapidly oscillating Fourier
integrand. It evaluates the same characteristic function, not a substitute
Black price or a payoff shortcut. The same original limit test and tolerances
remain. Scenario definitions, seeds, budgets, data splits and acceptance gates
are unchanged. v2 is separately frozen and must pass before any experiment.

# v3 domain coverage amendment, before controlled labels/fitting/training

v2 passed all 18 exact/PDE tests and completed validation selection on 800 quotes across 14 dates. Both Heston arms selected the 14% scenario; BS_TERM was selected. These observations remain validation, not a new final test. No final dates were downloaded/read.

An analytic range audit found that SMIRK_WEIGHTS can reach slow variance .06*.9*3=.162 > the original .12 training maximum, and fast variance .02*.1*.8=.0016 < the original .002 minimum. v3 widens slow-state bounds to [.00015,.18] and fast-state bounds to [.001,.22] before any teacher generation/training or baseline fit. This corrects coverage of unchanged scenarios; it is not an outcome-driven selection. Seeds, optimizer budgets, scenarios, bank, metrics, and fidelity thresholds remain unchanged. The same validation selection is replayed deterministically using the already obtained source ZIPs. v2 artifacts remain intact.


# v4 PINN objective amendment, before any lock, held-out fidelity test, controlled comparison or final-market access

v3 trained both seeds with the frozen recipe. The predeclared development-split check (teacher seed 93102; the held-out
fidelity seeds 93104/93105 untouched) failed all four frozen gates for the two-seed mean: price RMSE 1.33e-4 (gate 2e-5),
P95 2.94e-4 (5e-5), max 9.62e-4 (2e-4), IV RMSE 0.2004 vol pts (0.2). v3 was therefore not locked or evaluated. Its
checkpoints, development check and diagnosis are preserved in `../nifty_multifactor_v3/artifacts/`.

Diagnosis on training data only: training-set price RMSE 1.29e-4 matches development 1.33e-4, so the failure is underfitting,
not overfitting. At the final weights the price term is 1.5-2.0% of the objective and the PDE term 62-75%. The v3 price
normaliser (1e-3) is 50x looser than the acceptance RMSE gate (2e-5): a price error at the gate contributes 4e-4 to a loss of
order one, so the objective cannot target the protocol's own acceptance criterion. This is a defect in the objective scaling,
identified from training data, not an outcome-driven preference.

Change: each supervised term is normalised by its own acceptance tolerance, price MSE/(2e-5)^2 and IV MSE/(0.002)^2. The physics
terms keep their v3 absolute weights (0.1 x PDE MSE/0.01^2 and 0.1 x convexity). Architecture family, depth, seeds, teacher data,
sampling, collocation, batch sizes, learning rate, schedule shape, L-BFGS settings and all gates are unchanged.

Predeclared ladder, trained in order and selected on the development split only:
- C1: new normalisation, v3 budget (width 128, 10,000 Adam steps)
- C2: C1 with 40,000 Adam steps
- C3: C2 with width 256, depth still 5; trained only if C1 and C2 both fail
Lock the first candidate whose two-seed mean passes all four gates on development. If none passes, lock the lowest development
price RMSE; the frozen held-out fidelity test is then expected to fail and no structural claim is permitted. No other candidates,
seeds or settings are searched. C1 and C2 may train concurrently for wall-time only; the rule makes C2 irrelevant if C1 passes.

Because the development split now selects, it is no longer independent fidelity evidence. The untouched held-out fidelity
(seed 93104) and PDE-fidelity (seed 93105) tests inside the frozen `evaluate()` remain the acceptance test.

Inherited byte-identically from v3 with per-file SHA-256 verification (`run.py inherit`): scenario bank, market validation
sources and cleaned quotes, market selection (Single Heston 14%, Double Heston 14%, BS_TERM), controlled cases and all 50
surfaces, all 50 Single-Heston/Black-Scholes baselines (their records still cite the v3 manifest under which they ran), and the
teacher train/development/collocation data. Exact-pricer tests are rerun in v4. The final market interval remains unopened.
