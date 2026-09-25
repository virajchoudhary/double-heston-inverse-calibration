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
