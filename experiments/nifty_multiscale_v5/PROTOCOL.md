# Multiscale PINN extension and retrospective crisis reconstruction

Specified on 2026-09-18 before v5 training or calibration. Existing v4 results
and the earlier fixed-parameter crisis results have already been examined.
This is a development extension, not a prospectively untouched market test.

## Motivation and scope

Christoffersen, Heston and Jacobs (2009),
https://pure.au.dk/ws/files/17142435/rp09_34.pdf, motivate two variance factors
by independent variation of smile level/slope and different persistence. Their
1990–2004 S&P500 experiment is not a guarantee for crisis dates or NIFTY.
We examine all locally available eligible dates in the previously specified
March 2020, June 2025 and March 2026 windows, without selecting winning dates.
These are event windows, not a verified ranking of realized volatility.

Wang, Teng and Perdikaris (2021), https://arxiv.org/abs/2001.04536,
and Wang et al. (2023), https://arxiv.org/abs/2308.08468, motivate attention to
PINN loss scaling and multiscale optimization. The particular maturity gates
here are our proposed architecture, not a method claimed verbatim from them.
We do not claim to implement every research methodology.

## Architecture and ablation

Preserve the v4 C3 width-256, five-layer, two-seed models unchanged. Freeze their
weights and add three width-32, two-hidden-layer residual experts to the log-IV
correction. Add exact factor decay features exp(-kappa_i tau). Smooth sigmoid
gates on log maturity transition at 30 and 90 days (width 0.5); weights sum to
one. Zero-initialize expert heads, preserving the inherited function exactly.
Retain the Black variance ansatz, exact expiry payoff and call bounds. Convexity
is penalized and measured, not guaranteed. All PDE derivatives include gates.

Compare FROZEN, SHARED (same three experts and equal weights), and MULTISCALE
(smooth gates). SHARED and MULTISCALE use identical seeds, parameter counts,
batches, optimization steps and losses. This isolates the gate effect from
extra training and capacity; comparison to FROZEN alone cannot do so.
Both seeds 17 and 43 must be retained. No market-label weight updates.

Train on inherited synthetic training and collocation sets only: 800 Adam steps,
learning rate 2e-4 cosine-decayed to 1e-5, 512 labels and 64 PDE points per step;
then 60 L-BFGS iterations on the first 4096 training labels and 256 collocation
points. The inherited tolerance-scaled price, IV, PDE and convexity losses are
unchanged. Development is the inherited 4096 points. Choose the lowest ensemble
development price RMSE among candidates meeting all v4 fidelity gates and with
PDE RMSE no more than 10% above FROZEN on the first 256 existing collocation
points; retain FROZEN if no eligible extension improves price RMSE. No retries
with newly chosen hyperparameters after seeing results.

Freeze selection/checkpoint hashes before generating 4096 fresh synthetic
fidelity points (seed 185104) and 512 PDE points (185105). Report both extensions,
each seed, and the frozen baseline on that same set regardless of selection.
Re-evaluate old controlled surfaces by maturity; label this a reused diagnostic.
Synthetic teacher prices are Double Heston outputs, not neutral market truth.

## Retrospective market reconstruction

Reuse the hash-verified existing cleaned NIFTY panel and its entire-strike-pair
anchor/test split. Same-date carry and calibration use anchors only. Score all
eligible held-out quotes, including errors and boundary fits. No future-date
forecast claim; closes are not synchronous executable bid/ask quotes. Data
cleaning depends on market quote validity, as disclosed in the inherited audit.

Daily independently recalibrate (a) BS flat and BS per-expiry volatility; (b)
five-parameter exact Single Heston; (c) three-parameter DH PINN: common structural
variance scale and two initial states within its trained domain. BS family is
chosen on an inner anchor split (every fifth row), then refit on all anchors.
SH uses differential evolution and four local starts, max_nfev 150. PINN uses
four log-space LHS starts, max_nfev 150 and analytic gradients. No global optimum
is claimed. Save every optimizer status and boundary contact. Exact DH repricing
at the PINN-calibrated parameters separates network error from market error;
it is not an independently fitted ten-parameter DH benchmark. This restricted
PINN family and full SH have unequal flexibility; results cannot establish
general model-class superiority or unique structural parameter recovery.

Primary metric: square root of equally weighted daily mean squared errors in
OTM price/(D F). Also raw index-point RMSE, IV errors on jointly invertible
quotes, failures, 7–30 / 30–90 / 90–100 day buckets and paired daily errors.
Block bootstrap over ordered dates within each window, block length 5, 5000
replicates, seed 185119. Report 95% descriptive and 99.167% Bonferroni intervals
for six window-by-comparator differences (BS and SH in each of three windows).
Short retrospective windows and reused research data limit inferential claims.
No universal superiority guarantee is possible. Honest failures are results.
