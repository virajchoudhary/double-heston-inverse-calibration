# When Does a Second Variance Factor Pay? Pre-Registered Evidence from Five Cryptocurrency Option Markets

*(All prose below is original and written for this study. Numbers are taken directly from the
frozen experiment artifacts. Paste section by section; adapt headings to your target venue.)*

---

## ABSTRACT

Two-factor stochastic volatility models are widely believed to price options better than
one-factor models because they can represent a fast and a slow source of variance at the same
time. That belief is rarely tested in a way that could have falsified it. We report a
pre-registered comparison of Black–Scholes, the Heston model and the Double-Heston model on
16,186 option quotes drawn from five cryptocurrency markets on Deribit: Bitcoin, Ether, Solana,
XRP and Hyperliquid. Every evaluation date was selected from volatility-index or spot data
**before any option data was downloaded**, every model was calibrated separately on each date,
and accuracy was measured only on quotes and expiries that were withheld from calibration.
Significance required a one-sided Wilcoxon signed-rank test **and** a cluster-bootstrap lower
bound above zero, so that neither test alone could produce a result. The Double-Heston model
beat Black–Scholes on every asset and in nearly every regime, reducing median implied-volatility
error from 8.13 to 1.86 volatility points on Bitcoin shock dates. Against the one-factor Heston
model the picture is very different: Double Heston won decisively on Bitcoin and Ether, and
failed to win on Solana, XRP and Hyperliquid. The two markets where it won are the two that
quote six to seven expiries per day; the three where it did not quote about four. We argue that
the second variance factor only pays when the surface contains enough distinct maturities to
identify two mean-reversion speeds, and that the common practice of reporting only favourable
markets conceals this boundary. We also document a defect in our own Black–Scholes baseline,
the correction, and both sets of results.

**Keywords:** stochastic volatility, Double Heston, option pricing, cryptocurrency options,
model identifiability, pre-registration

---

## I. INTRODUCTION

Anyone pricing options has to choose how much structure to buy. The Black–Scholes formula
assumes one constant volatility and is wrong in a way every practitioner can see: implied
volatility varies with strike and with maturity. The Heston model replaces the constant with a
single mean-reverting variance process and captures much of that variation. A natural next step
is to add a second variance process running on a different time scale, which is what the
Double-Heston model does. The argument for it is intuitive. Short-dated options respond to
volatility that mean-reverts quickly; long-dated options respond to a slower component. One
process cannot be fast and slow at once, so two should fit better.

The argument is appealing, and the extra factor is not free. It doubles the number of variance
parameters from five to ten, and those parameters have to be recovered from a finite set of
quoted prices. If the quoted surface does not contain enough distinct maturities, the two
mean-reversion speeds cannot be told apart, and the extra flexibility buys nothing while making
calibration harder. Whether the trade is worth making is therefore an empirical question, and it
is a question about the **data**, not only about the model.

Most published comparisons answer it on one market, usually a major equity index, and report a
favourable result. That design cannot reveal where the advantage stops. Our study is built to
find the boundary rather than to demonstrate the advantage. We test the same three models on
five cryptocurrency markets that differ sharply in how many expiries they quote, and we fix
every choice that could bias the outcome before looking at any option price.

Our contributions are:

1. A pre-registered, five-market comparison of Black–Scholes, Heston and Double Heston, in which
   the evaluation dates were chosen from volatility or spot data before any option data was
   fetched, and the success criterion was fixed in advance.
2. Evidence that the Double-Heston advantage over Black–Scholes is large and universal across
   these markets, while its advantage over one-factor Heston is **conditional** and appears only
   where the surface quotes roughly six or more expiries per day.
3. An identifiability explanation for that boundary, together with the negative results that
   support it, including one market where the one-factor model is significantly better.
4. A full disclosure of a defect discovered in our own baseline implementation, the frozen
   correction, and both the flawed and corrected results.

We do not claim that Double Heston is the best available model, that these results transfer to
equity or foreign-exchange markets, or that they imply a trading strategy. We claim something
narrower and, we think, more useful: the value of a second variance factor depends on whether
the market gives you enough maturities to estimate it.

---

## II. BACKGROUND AND RELATED WORK

Black and Scholes [1] and Merton [2] give the constant-volatility benchmark, and Black [3]
adapts it to forward-settled contracts, which is the convention used for the instruments studied
here. The systematic failure of that benchmark — implied volatility depending on strike and
maturity — is documented at length by Gatheral [4].

Heston [5] introduced a correlated square-root variance process with a closed-form
characteristic function, which made calibration practical. Bakshi, Cao and Chen [6] compared the
main alternatives on equity index options and found stochastic volatility to be the single most
valuable addition. Bates [7] and Duffie, Pan and Singleton [8] set out the affine framework in
which multiple variance factors and jumps can be combined while preserving tractability.
Christoffersen, Heston and Jacobs [9] proposed the two-factor specification we use, showing that
two variance processes with different persistence reproduce the index smirk and its term
structure better than one. Our Double-Heston implementation follows their formulation.

Pricing under these models is usually done by Fourier inversion. Carr and Madan [10] introduced
the FFT approach and Fang and Oosterlee [11] the cosine expansion. Numerical care is required:
Lord and Kahl [12] show how naive complex-logarithm handling produces discontinuities, and Cui,
del Baño Rollin and Germano [13] give a stable formulation and gradient for calibration. The
Feller condition [14] governs whether the variance process can reach zero and is commonly
imposed or relaxed as a calibration variant; we test both.

Cryptocurrency option markets are newer and less studied. Madan, Reyners and Schoutens [15]
calibrated advanced models to Bitcoin options and found the data demanded heavier tails than
equity markets. Hou, Wang, Chen and Härdle [16] fitted a stochastic-volatility-with-jumps model
to Bitcoin options and reported substantial variance risk premia. Alexander and Imeraj [17]
document the microstructure of the Deribit venue and the behaviour of its volatility index, and
in later work [18] examine hedging under the observed smile. These studies establish that crypto
surfaces are steep, volatile, and thinner than equity surfaces — the conditions under which our
identifiability question becomes sharp.

On inference, we follow standard practice for clustered, non-normal data: the Wilcoxon
signed-rank test [19], the bootstrap [20], and cluster-level resampling as discussed by Cameron,
Gelbach and Miller [21]. On research design, our procedure is a direct application of
pre-registration [22] to model comparison: the outcome measure, the selection rule and the
success criterion were written down and hashed before the data were seen.

What is missing from this literature, in our reading, is a study designed to locate the
**limits** of a richer model rather than to demonstrate its advantages. That is the gap we
address.

---

## III. MODELS

All three models are priced in a forward-normalised, undiscounted convention. Writing `F` for
the forward price of the underlying at the option's expiry, `K` for the strike,
`x = log(F/K)` for log-forward moneyness and `tau` for time to maturity in years, we work with
the normalised call value `c = C/F`. This removes the discount factor and the level of the
underlying from the comparison and matches how the venue quotes its contracts.

### A. Black–Scholes with a term structure

The constant-volatility model is given one volatility per expiry rather than one volatility for
the whole surface. This is a deliberately generous baseline: it already absorbs the entire
maturity dimension of the smile and leaves only the strike dimension unexplained. A quote in a
calibrated expiry is priced with that expiry's volatility. A quote in an expiry that was
withheld is priced by interpolating total variance linearly between the neighbouring calibrated
expiries, and by a flat extension beyond them.

### B. Heston (one variance factor)

The underlying and its variance follow

    dF/F     = sqrt(v) dW1
    dv       = kappa (theta - v) dt + sigma sqrt(v) dW2,     corr(dW1, dW2) = rho

with five parameters `(kappa, theta, sigma, rho, v0)`. Here `kappa` is the speed at which
variance returns to its long-run level `theta`, `sigma` is the volatility of variance, `rho`
couples spot and variance shocks and produces the skew, and `v0` is the variance today.

### C. Double Heston (two variance factors)

Two independent square-root processes drive the same underlying:

    dF/F     = sqrt(v1) dW1 + sqrt(v2) dW3
    dv_i     = kappa_i (theta_i - v_i) dt + sigma_i sqrt(v_i) dW_{2i},   corr = rho_i

giving ten parameters. The intended division of labour is that one factor reverts quickly and
governs short-dated options while the other reverts slowly and governs long-dated ones. Because
the factors are independent, the characteristic function of the log-forward is the product of
the two single-factor characteristic functions, so the pricing cost is essentially twice that of
one-factor Heston.

### D. Numerical reference

Prices are obtained by Gauss–Laguerre inversion of the characteristic function using the
Little-Heston-Trap parameterisation to avoid branch-cut discontinuities [12]. A 128-node result
is checked against a 96-node result; if the two disagree by more than `1e-7`, or if a
no-arbitrage price bound is violated, the code falls back to independent adaptive quadrature
rather than clipping the value. Fallbacks are counted and reported.

---

## IV. DATA AND PRE-REGISTERED DESIGN

### A. Markets

We use Deribit, which lists the deepest cryptocurrency option books. Bitcoin and Ether options
are inverse contracts settled in the underlying coin; Solana, XRP and Hyperliquid options are
linear contracts settled in USDC. In both conventions the quoted price is an undiscounted
Black-76 value on the expiry forward, so the forward-normalised value `c` is directly
observable. Each quote's forward is recovered from the trade itself and the venue index, rather
than assumed.

### B. Date selection, fixed before any option data was fetched

This is the core of the design. For Bitcoin and Ether we used Deribit's DVOL volatility index.
A **shock onset** is a day on which DVOL exceeds 1.2 times its median over the previous 60 days,
de-duplicated so that onsets cannot fall within 20 days of each other; we then evaluate the
surfaces 1, 4 and 8 days after each onset. **Calm** dates are days on which the index sits within
5% of its 60-day median, at least 20 days away from any onset, with a sample drawn by a fixed
random seed.

Deribit publishes no volatility index for Solana, XRP or Hyperliquid. For those we used the
annualised standard deviation of the last ten daily log returns of the perpetual contract, with
a shock threshold of 1.5. Hyperliquid options have too short a history for an episode design, so
every calendar day in the listing window is an evaluation date.

Every one of these rules, including the thresholds, the de-duplication window, the offsets and
the random seeds, was written to a configuration file and hashed **before any option surface was
downloaded**. The manifests record `option_data_fetched_before_freeze: false`. Dates for which
data turned out to be unavailable are disclosed and were never replaced by other dates.

### C. Calibration and held-out evaluation

Each model is calibrated independently on each date, minimising a vega-weighted squared pricing
error, with twelve restarts for the Heston models to reduce the risk of a local optimum. Two
held-out designs were fixed in advance:

* **Design A** withholds a random subset of quotes from every expiry.
* **Design B** withholds **entire expiries**, so the model must price a maturity it has never
  seen. Design B is the primary endpoint, because it tests the maturity structure directly,
  which is exactly where a second variance factor is supposed to help.

Accuracy is reported as the root-mean-square error in implied volatility points on the held-out
quotes. We report implied volatility rather than price because it is comparable across strikes
and maturities.

### D. Success criterion, fixed in advance

Individual dates are not independent: several dates can belong to the same volatility episode.
We therefore cluster by episode (or by ISO week for Hyperliquid) and require **both** of the
following before declaring that one model beats another:

1. a one-sided Wilcoxon signed-rank test over clusters with `p < 0.05`, **and**
2. a cluster-bootstrap 95% lower bound on the mean paired difference that is strictly above
   zero, using 5,000 replicates and a fixed seed.

Requiring both is not a formality. As reported below, it prevented three results that pass the
Wilcoxon test alone from being declared wins.

### E. Disclosure of a defect in our own baseline

After the Bitcoin results were produced we found a defect in our Black–Scholes implementation.
Because each quote's maturity was computed from its own trade timestamp, quotes in the same
expiry had marginally different maturities, and the routine that placed one volatility per
expiry in fact placed one volatility **per quote**. The baseline therefore reproduced its own
calibration quotes exactly and interpolated between individual quotes rather than between
expiries.

We wrote and hashed the correction *before* running it, changed only the grouping, left the
original outputs in place as a permanent record, and report both. The correction makes
Black–Scholes **stronger**, improving its median held-out error on Bitcoin shock dates from 9.63
to 8.13 volatility points. Every Black–Scholes number in this paper is the corrected one. The
Heston fits do not use knots and were unaffected.

---

## V. RESULTS

### A. Double Heston against Black–Scholes

The two-factor model beats the constant-volatility baseline on every asset, in every regime with
enough clusters to test, and by a wide margin. On Bitcoin shock dates the median held-out error
falls from 8.13 to 1.86 volatility points, a reduction of about 77%. The paired difference is
positive on 29 of 29 dates and in 10 of 10 episodes, with a cluster-bootstrap interval of
[5.22, 8.31] volatility points.

This is the expected result and we report it mainly to establish that the pipeline works and the
markets behave sensibly. It is not the interesting finding.

### B. Double Heston against one-factor Heston

Here the five markets divide cleanly into two groups (Table II).

On **Bitcoin** the two-factor model wins in both regimes. On shock dates the mean advantage is
0.57 volatility points, positive in 23 of 29 dates and in all 10 episodes, with `p = 0.001` and a
bootstrap interval of [0.39, 0.71]. On calm dates the advantage is larger, 1.11 points, with an
interval of [0.64, 1.65].

On **Ether** the two-factor model also wins in both regimes: 0.96 points on shock dates
(interval [0.61, 1.46]) and 0.53 points on calm dates (interval [0.22, 0.86]).

On **Solana, XRP and Hyperliquid** it does not win in any cell. The failures take three different
forms, and all three are informative:

* **Solana, calm dates.** The one-factor model is *significantly better*: the mean difference is
  −0.48 volatility points with an interval of [−0.98, −0.02] lying entirely below zero.
* **XRP, shock dates.** The Wilcoxon test passes (`p = 0.037`) but the bootstrap interval
  [−0.08, 1.82] contains zero, so the pre-registered criterion is not met.
* **Hyperliquid, all dates.** The same pattern: `p = 0.029` with an interval of [−0.06, 0.74].

The last two cases are exactly why we required two tests. A single test would have produced two
additional "wins" that the data do not support.

### C. The pattern behind the split

The division is not random with respect to market structure. Table I reports the median number
of distinct expiries quoted per evaluation date. Bitcoin quotes seven, Ether six; Solana, XRP and
Hyperliquid quote about four. The two markets where the second variance factor pays are precisely
the two that quote the most maturities.

This is what identifiability predicts. Two mean-reversion speeds can only be separated if the
data contain maturities at which their effects differ. With four expiries, and often with two or
three of those clustered within a month, a fast and a slow factor produce nearly the same fitted
surface, and the second factor adds parameters without adding explanatory power. With six or
seven expiries spanning days to several months, the two time scales leave distinguishable
fingerprints and the extra factor earns its place.

### D. Illustration on a single surface

Figure 1 shows Bitcoin call prices against the underlying on 27 July 2024, with all three models
calibrated to that date and evaluated on the traded moneyness range. The curves are close
because the option payoff dominates the price, but Single Heston lies above the quotes,
Black–Scholes below them, and Double Heston passes through them. On the twenty-day expiry the
Double-Heston price error is 4.7 times smaller than the better of the two baselines. Figure 2
repeats the comparison against time to maturity and shows the characteristic step in the
Black–Scholes curve where it switches between expiry knots — a structural consequence of giving
it one volatility per expiry.

We stress that a single date is an illustration, not evidence. The evidence is the pre-registered
aggregate in Table II.

---

## VI. DISCUSSION

The headline result of most model-comparison papers is that the richer model wins. Ours is that
the richer model wins **when the data can identify it**, and that this condition is not satisfied
in three of the five markets we examined.

We think this reframing matters for three reasons.

First, it is actionable. A practitioner deciding whether to move from one to two variance factors
can look at the expiry count of the surface they actually trade, rather than at a published
result obtained on a different and richer market.

Second, it explains conflicting results in the literature without needing to attribute them to
implementation differences. Studies on major equity indices, where dozens of expiries are listed,
should find a two-factor advantage. Studies on thin surfaces should not. Both can be right.

Third, it suggests that the useful question is not "which model is better" but "how much
structure can this surface support". A model that cannot be identified is not a better model
whose benefits are hidden by noise; it is an over-specified model whose extra parameters are
absorbing noise.

The Solana calm-date result, where the one-factor model is significantly better, is consistent
with this reading. Over-specification does not merely fail to help, it can actively hurt when the
extra freedom fits features of the calibration set that do not recur in the held-out set.

---

## VII. LIMITATIONS

We list the limitations that we consider material.

The study covers cryptocurrency options on a single venue. Crypto surfaces are steeper and
thinner than equity surfaces, so the specific expiry threshold we observe, roughly six per day,
should not be transferred to other asset classes without re-testing. The mechanism —
identifiability of two mean-reversion speeds — should transfer; the numerical threshold need not.

Expiry count is not the only difference between these markets. Bitcoin and Ether are also older,
more liquid, and have longer option histories than the other three. We cannot fully separate
maturity coverage from liquidity with five markets, and we do not claim to. Expiry count is the
variable that identifiability theory singles out, and it orders the results without exception,
but a larger cross-section would be needed to isolate it.

Hyperliquid has only four shock dates in one episode, so that cell is uninformative and no
conclusion should be drawn from it.

Our calibration bounds bind in a substantial minority of fits: the volatility-of-variance upper
bound is reached in 63 of 132 Heston fits and in about 80 of 132 Double-Heston fits. Relaxing
those bounds after seeing the results is not permitted under our own protocol, so we record it
as the leading candidate improvement for a future, independently pre-registered experiment.

Finally, we measure pricing accuracy on held-out quotes from the same day. We do not test
forecasting, hedging performance, or economic value, all of which are separate questions.

---

## VIII. CONCLUSION

We compared Black–Scholes, Heston and Double Heston on 16,186 option quotes from five
cryptocurrency markets, under a design fixed and hashed before any option data was seen. The
two-factor model beat the constant-volatility baseline everywhere, cutting median held-out
implied-volatility error on Bitcoin shock dates from 8.13 to 1.86 volatility points. Against the
one-factor model it won decisively on Bitcoin and Ether and failed to win on Solana, XRP and
Hyperliquid, with one market in which the simpler model was significantly better.

The two markets in which the second variance factor paid are the two that quote six to seven
expiries per day; the three in which it did not quote about four. We read this as an
identifiability boundary rather than a property of the model: two mean-reversion speeds can only
be estimated from a surface that contains enough distinct maturities to separate them.

The practical implication is that the choice between one and two variance factors should be made
by looking at the surface, not only at the model. The methodological implication is that
pre-registering the selection rule and requiring two independent significance criteria changed
our conclusions — three results that would have been reported as wins under a single test did not
survive both.

---

## TABLE I — MARKETS AND DATA

| Asset | Contract | Evaluation dates | Held-out quotes | Median expiries per day | Period |
|---|---|---:|---:|---:|---|
| Bitcoin (BTC) | Inverse, coin-settled | 66 | 6,171 | 7 | May 2021 – Jul 2026 |
| Ether (ETH) | Inverse, coin-settled | 53 | 3,619 | 6 | May 2021 – Jul 2026 |
| Solana (SOL) | Linear, USDC | 32 | 2,687 | 4 | Nov 2024 – Aug 2026 |
| XRP | Linear, USDC | 26 | 1,249 | 4 | Oct 2024 – Aug 2026 |
| Hyperliquid (HYPE) | Linear, USDC | 55 | 2,460 | 4 | Jun 2026 – Sep 2026 |
| **Total** | | **232** | **16,186** | | |

---

## TABLE II — HELD-OUT ACCURACY AND PRE-REGISTERED TESTS (DESIGN B)

Median implied-volatility RMSE in volatility points. "DH wins" requires Wilcoxon `p < 0.05`
**and** a cluster-bootstrap 95% lower bound above zero.

| Asset | Regime | Dates | Clusters | BS | SH | DH | SH−DH mean | *p* | Bootstrap 95% | DH beats SH |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|:---:|
| BTC | shock | 29 | 10 | 8.13 | 2.50 | **1.86** | +0.57 | 0.001 | [0.39, 0.71] | **yes** |
| BTC | calm | 21 | 21 | 6.40 | 2.53 | **1.37** | +1.11 | 0.0001 | [0.64, 1.65] | **yes** |
| ETH | shock | 27 | 10 | 8.76 | 3.23 | **2.32** | +0.96 | 0.001 | [0.61, 1.46] | **yes** |
| ETH | calm | 26 | 26 | 8.77 | 2.68 | **2.34** | +0.53 | 0.0026 | [0.22, 0.86] | **yes** |
| SOL | shock | 16 | 7 | 10.15 | 3.82 | 3.67 | +0.27 | 0.234 | [−0.10, 1.14] | no |
| SOL | calm | 16 | 16 | 6.98 | **2.42** | 2.36 | −0.48 | 0.839 | [−0.98, −0.02] | no (SH better) |
| XRP | shock | 18 | 9 | 13.09 | **5.24** | 5.72 | +0.59 | 0.037 | [−0.08, 1.82] | no |
| XRP | calm | 8 | 8 | 7.00 | **3.28** | 3.52 | −0.19 | 0.727 | [−0.71, 0.30] | no |
| HYPE | all | 55 | 13 | 4.68 | 3.61 | **3.26** | +0.27 | 0.029 | [−0.06, 0.74] | no |

Against Black–Scholes, Double Heston wins in every cell above with sufficient clusters, with
bootstrap intervals from [1.41, 6.99] on Hyperliquid to [5.22, 8.31] on Bitcoin shock dates.

---

## FIGURE CAPTIONS

**Figure 1.** Bitcoin call prices against the underlying, 27 July 2024, at four expiries. All
three models are calibrated to that date. Prices are divided by the strike so that quotes across
strikes are comparable, and the horizontal range is restricted to the region in which options
carry meaningful time value. Circles are market quotes. Single Heston lies above the quotes and
Black–Scholes below them; Double Heston passes through them. Two of the four expiries were
withheld from calibration entirely.

**Figure 2.** Bitcoin call prices against time to maturity at fixed moneyness, same date.
Maturity decreases from left to right, so each curve decays towards its payoff at expiry. The
step in the Black–Scholes curve is the point at which it switches between expiry knots: with one
volatility per expiry it interpolates total variance piecewise and cannot produce a smooth term
structure.

**Figure 3.** Distribution of absolute implied-volatility error across all held-out Bitcoin
quotes, by model. The empirical cumulative distribution of the Double-Heston error lies to the
left of the Single-Heston error at every quantile, which is a stronger statement than a
difference in means.

**Figure 4.** Per-date mean advantage of Double Heston over Single Heston across the held-out
Bitcoin test dates, with 95% intervals, ranked. The shaded band is the cluster-bootstrap interval
for the overall mean. Dates on which the one-factor model was better are shown rather than
omitted.

---

## REFERENCES

[1] F. Black and M. Scholes, "The pricing of options and corporate liabilities," *Journal of
Political Economy*, vol. 81, no. 3, pp. 637–654, 1973.

[2] R. C. Merton, "Theory of rational option pricing," *Bell Journal of Economics and Management
Science*, vol. 4, no. 1, pp. 141–183, 1973.

[3] F. Black, "The pricing of commodity contracts," *Journal of Financial Economics*, vol. 3,
no. 1–2, pp. 167–179, 1976.

[4] J. Gatheral, *The Volatility Surface: A Practitioner's Guide*. Hoboken, NJ: Wiley, 2006.

[5] S. L. Heston, "A closed-form solution for options with stochastic volatility with
applications to bond and currency options," *Review of Financial Studies*, vol. 6, no. 2,
pp. 327–343, 1993.

[6] G. Bakshi, C. Cao, and Z. Chen, "Empirical performance of alternative option pricing models,"
*Journal of Finance*, vol. 52, no. 5, pp. 2003–2049, 1997.

[7] D. S. Bates, "Post-'87 crash fears in the S&P 500 futures option market," *Journal of
Econometrics*, vol. 94, no. 1–2, pp. 181–238, 2000.

[8] D. Duffie, J. Pan, and K. Singleton, "Transform analysis and asset pricing for affine
jump-diffusions," *Econometrica*, vol. 68, no. 6, pp. 1343–1376, 2000.

[9] P. Christoffersen, S. Heston, and K. Jacobs, "The shape and term structure of the index
option smirk: Why multifactor stochastic volatility models work so well," *Management Science*,
vol. 55, no. 12, pp. 1914–1932, 2009.

[10] P. Carr and D. B. Madan, "Option valuation using the fast Fourier transform," *Journal of
Computational Finance*, vol. 2, no. 4, pp. 61–73, 1999.

[11] F. Fang and C. W. Oosterlee, "A novel pricing method for European options based on
Fourier-cosine series expansions," *SIAM Journal on Scientific Computing*, vol. 31, no. 2,
pp. 826–848, 2008.

[12] R. Lord and C. Kahl, "Complex logarithms in Heston-like models," *Mathematical Finance*,
vol. 20, no. 4, pp. 671–694, 2010.

[13] Y. Cui, S. del Baño Rollin, and G. Germano, "Full and fast calibration of the Heston
stochastic volatility model," *European Journal of Operational Research*, vol. 263, no. 2,
pp. 625–638, 2017.

[14] W. Feller, "Two singular diffusion problems," *Annals of Mathematics*, vol. 54, no. 1,
pp. 173–182, 1951.

[15] D. B. Madan, S. Reyners, and W. Schoutens, "Advanced model calibration on bitcoin options,"
*Digital Finance*, vol. 1, pp. 117–137, 2019.

[16] A. J. Hou, W. Wang, C. Y. H. Chen, and W. K. Härdle, "Pricing cryptocurrency options,"
*Journal of Financial Econometrics*, vol. 18, no. 2, pp. 250–279, 2020.

[17] C. Alexander and A. Imeraj, "The Bitcoin VIX and its variance risk premium," *Journal of
Alternative Investments*, vol. 23, no. 4, pp. 84–109, 2021.

[18] C. Alexander and A. Imeraj, "Delta hedging bitcoin options with a smile," *Quantitative
Finance*, vol. 23, no. 5, pp. 799–817, 2023.

[19] F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1,
no. 6, pp. 80–83, 1945.

[20] B. Efron, "Bootstrap methods: Another look at the jackknife," *Annals of Statistics*,
vol. 7, no. 1, pp. 1–26, 1979.

[21] A. C. Cameron, J. B. Gelbach, and D. L. Miller, "Bootstrap-based improvements for inference
with clustered errors," *Review of Economics and Statistics*, vol. 90, no. 3, pp. 414–427, 2008.

[22] B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor, "The preregistration
revolution," *Proceedings of the National Academy of Sciences*, vol. 115, no. 11,
pp. 2600–2606, 2018.
