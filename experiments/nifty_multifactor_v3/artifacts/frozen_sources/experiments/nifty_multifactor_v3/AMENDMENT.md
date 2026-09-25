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

