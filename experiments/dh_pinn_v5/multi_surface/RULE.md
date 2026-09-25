# Surface selection rule (fixed before any model error was computed)

**Question.** On which of today's option surfaces should a second variance timescale matter most?

**Model-free score.** For each surface, take the market at-the-money implied volatility of every expiry,
convert to total variance w(τ) = σ_atm(τ)² τ, and fit the **single-timescale** (Heston) form

    w(τ) = θ τ + (v₀ − θ) (1 − e^(−κτ)) / κ ,   κ > 0

by least squares over (θ, v₀, κ). The score is

    two_timescale_score = RMSE(fit residual) / mean(w)

A surface a single mean-reversion rate explains has a score near zero; a surface needing two
separated timescales scores high. Nothing about Black-Scholes, Single Heston, Double Heston or
the PINN enters this score.

**Selection.** Among surfaces that are inside the PINN's trained level range (fitted scale strictly
inside [0.7, 3.2]) and have at least 6 usable expiries, take the **highest score**. That surface is
used for the two requested graphs.

**Reporting.** Every surface's score and every model's error are reported, win or lose. If Double
Heston does not win on the selected surface, that is the result.

**Filters** (identical to hardcoded_v1): mid of bid/ask with bid > 0, ask > bid, ask ≤ 3·bid;
out-of-the-money only; 7–730 days; |log(F/K)| ≤ 0.36; ≥ 6 quotes per expiry; forward and discount
per expiry from a put-call parity regression with R² ≥ 0.999.
