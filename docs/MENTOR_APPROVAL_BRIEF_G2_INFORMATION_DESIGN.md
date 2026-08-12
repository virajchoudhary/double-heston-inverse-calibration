# Mentor Approval Brief: G2 Information Design

## Current result

The validated Double Heston pricing engine works, but optimizer engineering did not resolve NTPC parameter instability: the final optimizer-cap classification is `OPTIMIZER_CAP_UNRESOLVED`, and optimizer-only work is closed. Adding three-date temporal information reduced materially displaced starts from `11` to `7`, clusters from `7` to `3`, median separation from `0.399516908` to `0.324066116`, and maximum separation from `0.627751647` to `0.481226608`. Boundary-hit rate remained `1.0`, however, and 15-Jul holdout RMSE worsened from `0.926824720` to `0.976300061` (`+5.338%`). The result is `MULTI_DATE_INSUFFICIENT`; `G2 = NOT_PASSED`, so the final representation remains unfrozen.

## Approval request A: formal G2 safeguard

Before final representation freeze, require both a market-supported representation **and** sufficient ten-parameter informativeness/stability, or an explicitly mentor-approved revised formulation. This safeguard does not alter the canonical ten-parameter Double Heston model.

## Approval request B: one bounded richer NTPC-only study

Authorize one predeclared information-density study using candidate development dates 01-Jul, 08-Jul, 15-Jul, 22-Jul, and 29-Jul, retaining only dates that pass the existing official-NSE activity/support contract. 08-Jul and 29-Jul have already been used for Stage A market-support analysis but have not been used in the richer calibration treatment; none of these five dates is eligible for final G8.

Keep fixed the canonical ten parameters, production pricer, bounds, constraints, optimizer, objective, quote rules, and shared-structure/date-specific-variance concept. Change only temporal information density. Exclude priors, regularization, temporal smoothing, realized-volatility supervision, CIR penalties, wider bounds, a new optimizer, a new sector, ANN, and PINN.

The inherited separation and holdout rules remain fixed. Any new minimum eligible-date count, minimum rows/date, boundary-hit threshold, or `RICHER_INFORMATION_*` mapping is **PROPOSED — REQUIRES MENTOR CONFIRMATION** and must be frozen before results are seen.

## Mentor decision

`G2 safeguard: APPROVE / REJECT`

`One bounded richer NTPC study: APPROVE / REJECT`

`Mentor notes: __________________`
