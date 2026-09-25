# NIFTY 50: fixed-parameter Single vs Double Heston PINNs

Completed 17 September 2026. This is a separate forward-pricing experiment;
the original inverse-calibration experiments and model files are unchanged.

## Conclusion

**This trial does not justify a meaningful or consistent Double Heston edge.**
The neural networks accurately approximate their fixed models, but both fixed
models fit the observed NIFTY option prices poorly. A good neural approximation
is not the same as a good market model.

Independent numerical pricing slightly favors Double Heston in June 2025 and
Single Heston in March 2020 and March 2026. The differences are extremely small
relative to market error. In March 2026 the PINN comparison even reverses the
reference-engine ordering. Do not use that neural-only result as evidence of a
two-factor advantage, and do not promote these parameters as a final NIFTY fit.

## Exact scope and data

| Predeclared window | Trading dates | Held-out OTM quotes | Retained anchor diagnostic quotes |
|---|---:|---:|---:|
| March 2020 | 21 | 636 | 1,272 |
| June 2025 | 21 | 1,643 | 3,280 |
| February 28–March 31, 2026 | 19 | 1,624 | 3,237 |
| Total | 61 | 3,903 | 7,789 |

These are NIFTY index options, not ADANIPOWER, individual power-sector stocks,
NIFTY futures, or a synthetic market dataset. All three complete calendar
windows were retained; dates were not selected by which model won. The windows
are event-related samples, not a measured ranking of the most volatile days.

The source is 61 archived NSE F&O bhavcopy ZIPs, each checked against the saved
source-manifest SHA-256 before use. Original NSE URLs and hashes are retained
in `data_audit.json`. This verifies the retained local archive, not a fresh
remote re-download or a guarantee that every exchange closing quote is error-free.
The market file is separate from the generated synthetic training files.

Cleaning, identical for both models:

- Select legacy `OPTIDX/SYMBOL=NIFTY` or UDiFF `IDO/TckrSymb=NIFTY`.
- Use `CLOSE/ClsPric`, never substitute settlement prices. Require finite,
  positive strike, close, OHLC, trading volume and open interest; close and open
  must lie within the day's low/high. Remove identical duplicates; conflicting
  contract keys hard-stop. No identical duplicates were found.
- Preserve legacy `EXPIRY_DT` or UDiFF `FininstrmActlXpryDt` as actual expiry.
  Compute actual days/365, retaining 7–100 days (0.019178–0.273973 years).
  Never impose Tuesday/Thursday or round an expiry to one/two/three months.
- For each date/expiry require at least six traded call/put strike pairs. Sort
  strikes and reserve every third entire pair (index modulo 3 equals 1).
  Estimate forward F and discount D from the other pairs only, using robust
  `C-P = D*(F-K)` regression. Require positive F, discount corresponding to
  rates between -10% and 25%, anchor residual RMSE/F no more than 2.5%, and
  anchor strike span/F at least 4%. These are screening bounds, not assumed rates.
- Choose the OTM call when K >= F and the OTM put otherwise. Require
  `abs(log(F/K)) <= .35`, an invertible arbitrage-admissible Black price,
  IV between .03 and 2.5, and Black vega/(D*F) at least .002. Retain dates with
  at least two usable held-out quotes. Neither model's error enters cleaning.
- Record rejected rows, expiry-group failures and unavailable manifest dates.
  The retained anchor counts above are after IV/moneyness filters; carry was
  estimated using all anchor pairs that passed the preceding basic filters.

Legacy files do not supply underlying spot. No spot or dividend yield was
fabricated: this experiment uses anchor-inferred F and D and forward-normalized
prices. NIFTY is an index, so individual-stock split adjustments are not applied
to its option strikes/prices. Raw levels and source files are preserved.
Constituent changes are embedded in the published index, not reverse-engineered.

Limitations: daily closes are not synchronous executable bid/ask midpoints;
positive volume/OI cannot prove freshness. This is a screened close-price test,
not a complete microstructure/no-arbitrage certification of the market surface.
Missing manifest dates are explicitly listed, not filled with invented prices.

## Hard-coded parameters

Parameter order per factor: kappa (mean reversion), theta (long-run variance),
sigma (volatility of variance), rho (correlation), v0 (initial variance).
Variances are not volatilities and must not be squared again.

| Parameter | Double: slow | Double: fast | Matched Single |
|---|---:|---:|---:|
| kappa | 0.9 | 1.2 | 0.9483870967741935 |
| theta | 0.1 | 0.15 | 0.25 |
| sigma | 0.36 | 0.2 | 0.3124099870362662 |
| rho | -0.5 | -0.5 | -0.4847110454602491 |
| v0 | 0.36 | 0.2 | 0.56 |

Double Heston is published **DH1, Table 2** in
[Kyriakou, Brignone & Fusai (2024), Unified moment-based modelling of integrated stochastic processes](https://openaccess.city.ac.uk/id/eprint/29416/1/manuscript_cro.pdf).
This is a published numerical example, **not research-estimated NIFTY parameters**.
Single Heston is a study-defined deterministic reduction, not a separate claim
of published parameter values. With V=sum(v_i), Theta=sum(theta_i):

```
v0_single    = V
theta_single = Theta
kappa_single = sum(kappa_i*(v_i-theta_i)) / (V-Theta)
sigma_single = sqrt(sum(sigma_i^2*v_i) / V)
rho_single   = sum(rho_i*sigma_i*v_i) / (sigma_single*V)
```

This matches initial total variance, long-run mean variance, initial variance
drift, instantaneous variance diffusion and return–variance covariance. It does
not match the entire distribution or term structure. Both arms satisfy strict
Feller inequalities and economic bounds. The initial annualized volatility is
sqrt(.56), about 74.8%, in both arms. No parameter or initial variance is reset
to the observed daily market IV.

Only this one predeclared paired scenario was tested here. It is not a search
over every proposed parameter vector or evidence of globally optimal parameters.
DH1's two mean-reversion speeds are relatively close, which also limits what
this example can demonstrate about strongly separated volatility timescales.

## PINN and training

All five/ten model parameters are fixed. Only neural weights are optimized.
This is **forward reconstruction, not inverse parameter calibration**.

The existing regular PyTorch PINN is reused: five hidden layers of 128 tanh
neurons, float64, with a positive implied-variance correction to an analytic
Black-call price expression. It uses the project's engineered variance/moneyness
features, not a newly introduced Fourier-feature/Softplus-price architecture.
The terminal payoff is exact by construction. This is a teacher-assisted PINN,
not a PDE-only network: independent Fourier prices supply synthetic labels.
No exact-pricer parameter optimizer or market parameter polish is used.

Inputs are `[log(F/K), v, tau]` for Single and
`[log(F/K), v_slow, v_fast, tau]` for Double. Fixed structural coefficients are
passed separately. Varying variance **states** at PDE collocation points does
not change the fixed v0 used to price the market observations.

For each model and each seed (17 and 43):

- 2,048 synthetic training coordinates: Latin hypercube, uniform x in
  [-.35,.35] and tau in [7/365,100/365], evaluated at fixed initial variances.
  Same x/tau coordinates across arms. Multiple continuous maturities, not only
  monthly expiries. Labels: normalized call price and Black implied volatility.
- **18,000 distinct PDE collocation points**, with x/tau over the same domain,
  total variance in [.04,1.0]. Double splits total variance with a sampled
  slow-factor share in [.25,.75]; Single uses the same total-variance domain.
- 2,000 Adam steps (learning rate .001; label batch 256, PDE batch 192), then
  up to 150 L-BFGS iterations on fixed 512-label/512-collocation subsets.
  Identical budgets, architecture width/depth and float precision across arms;
  Double has more input features and a more expensive PDE, so wall time is not
  identical. Actual training times were approximately 55–76 seconds per run.
- Final checkpoints only; no best seed/epoch selected on market results.
  The reported PINN prediction is the arithmetic mean of the two final prices.

Loss:

```
L = (MSE(normalized_call, synthetic_call)
     + MSE(IV, synthetic_IV)
     + MSE(scaled_price_PDE_residual, 0)) / 0.01^2
    + 0.1 * mean(relu(-convexity_diagnostic)^2)
```

The residual contains x/variance second derivatives and x–variance mixed
derivatives. It is the price PDE residual divided by
`Black_call_derivative_wrt_total_variance * (total_variance + expected_average_variance)`.
It is not a raw currency-price residual. Feller and ordering penalties are not
optimized here: the fixed values already satisfy them. There is no parameter
target/recovery loss. Put prices are recovered by exact put–call parity without
clipping bad predictions to improve scores.

## Numerical verification versus market performance

Fresh synthetic testing uses 2,048 different x/tau coordinates and 2,048 new
PDE collocation points. All four runs passed the predeclared fidelity gates:
synthetic IV RMSE <= .002, market-coordinate PINN/reference price RMSE/(D*F)
<= .0002, and fresh scaled PDE RMSE <= .01.

| Model / seed | Synthetic IV RMSE | PINN/reference price RMSE/(D*F) | Fresh scaled PDE RMSE |
|---|---:|---:|---:|
| Single / 17 | .00015066 | .00001433 | .006295 |
| Single / 43 | .00024387 | .00001481 | .006473 |
| Double / 17 | .00021771 | .00002134 | .008181 |
| Double / 43 | .00018115 | .00001121 | .004820 |

96/128-node Fourier prices agree to within 1.9e-14 on the evaluated model
grids. A separately implemented adaptive-integration Double Heston engine also
passes nine corner/interior test cases. The transformed PDE residual was checked
against direct price automatic differentiation for both models.

These checks establish numerical accuracy within stated tolerances, not perfect
mathematical solutions or good market parameters. The tolerances are wider than
the very small difference between these particular model price surfaces.

### Market results

Primary metric: first average squared OTM price error/(D*F)^2 within each date,
then average dates equally and take the square root. The following percentages
are relative to the discounted index forward, **not percentage errors relative
to the option premium**.

| Window | Independent Single NRMSE | Independent Double NRMSE | DH winning dates, reference | Single PINN NRMSE | Double PINN NRMSE |
|---|---:|---:|---:|---:|---:|
| March 2020 | 2.943554% | 2.943659% | 2/21 | 2.943032% | 2.943095% |
| June 2025 | 6.329074% | 6.329010% | 21/21 | 6.328509% | 6.328023% |
| March 2026 | 6.210971% | 6.211018% | 3/19 | 6.210272% | 6.210054% |

Independent quote-weighted price RMSEs in index points, Single / Double:
303.912 / 303.920; 1568.360 / 1568.344; 1442.195 / 1442.206.
The absolute differences are below .02 index points, while model errors are
hundreds or thousands of index points. Averaged PINN/reference discrepancies
are about .11–.41 index points: enough to swamp the tiny model-ranking margin.

All date-level errors, individual seed predictions, IV errors and 95% paired
three-day moving-block bootstrap intervals are retained in the output files.
The intervals are conditional/descriptive, not causal war effects or
multiple-testing-adjusted proof. A tiny statistically signed difference is not
automatically an economically useful improvement.

### Plot interpretation

`crisis_price_comparison.png`: almost identical bars show that neither model
has a meaningful price-recovery advantage for this fixed-vector scenario.
Low neural/reference discrepancy does not remove the large market mismatch.

`crisis_iv_diagnostic.png`: the predicted IVs remain near 72–75% while observed
IV changes substantially. Both fixed models overprice most 2025/2026 quotes.
The dashed diagonal denotes perfect agreement; it is not a fitted model line.
These are close-implied volatilities, not observed instantaneous/realized volatility.

## Leakage, reproducibility and honest limits

- Protocol, parameters, filters and implementation hashes were saved before
  fitting or market model-error evaluation. Training opens no market quote file.
- No held-out price is used for neural training, structural fitting or forward/
  discount regression. Quote quality/availability are used for eligibility;
  this is not a claim that preprocessing never reads held-out quotes.
- Identical synthetic training/test coordinates: none. All 18,000 collocation
  rows per run are finite and unique. Raw and predicted contract keys are unique.
- Five new checks plus the two existing carry-isolation checks and existing
  Fourier regression check pass: **8 tests total**. This is evidence against
  specified failure modes, not a guarantee that every possible bug is absent.
- This is same-date surface reconstruction using anchor-derived carry, not
  future-date forecasting. Historical research context has already been seen;
  it is not described as a prospectively untouched blind market trial.
- No market quote fabrication, favorable-date filtering, best-seed selection,
  parameter tuning to favor Double, or post-result filter changes were performed.
- Evaluation emitted a read-only NumPy-view warning when creating a tensor;
  those inputs were only read, all outputs were finite, and no frozen code or
  setting was changed to suppress the warning.

The next useful research step would be a **newly declared**, market-plausible
parameter-scenario study (including genuinely separated factor timescales),
with both models treated comparably and all outcomes retained. It must not be
described as a continuation of this unchanged frozen test or a search until
Double Heston wins. Better PINN accuracy alone cannot fix inappropriate fixed
parameters. This experiment supplies no evidence of recovering true market
parameters, whose values are unknown.

## Reproduction and handoff

Working directory: `double-heston-v2-controlled`.

```
python scripts/mentor_dh_pinn/nifty_fixed_crisis.py prepare --out outputs/NEW_EMPTY_RUN
python scripts/mentor_dh_pinn/nifty_fixed_crisis.py train --out outputs/NEW_EMPTY_RUN --model single --seed 17
python scripts/mentor_dh_pinn/nifty_fixed_crisis.py train --out outputs/NEW_EMPTY_RUN --model single --seed 43
python scripts/mentor_dh_pinn/nifty_fixed_crisis.py train --out outputs/NEW_EMPTY_RUN --model double --seed 17
python scripts/mentor_dh_pinn/nifty_fixed_crisis.py train --out outputs/NEW_EMPTY_RUN --model double --seed 43
python scripts/mentor_dh_pinn/nifty_fixed_crisis.py score --out outputs/NEW_EMPTY_RUN
python -m pytest -q tests/test_nifty_fixed_crisis.py tests/test_crisis_option_coverage.py tests/test_torch_pricer.py
```

Do not overwrite this completed run. Existing directories deliberately fail.
Execution requires the archived NSE files under the parent workspace's `raw/`
and its `outputs/019fc8a0/nse_source_manifest.csv`, plus the existing Python
environment (PyTorch 2.13.0, NumPy, SciPy, pandas, matplotlib and pytest).

Output directory: `outputs/nifty_fixed_crisis_20260917/`.
Key files: `protocol.json`, `data_audit.json`, `clean_quotes.csv`,
`expiry_group_audit.csv`, `rejections.csv`, `test_predictions.csv`,
`daily_metrics.csv`, `results.json`, two PNG plots, and four model directories
containing fixed parameters, training inputs, history and `weights.pt`.

Protocol SHA-256: `5825aac20cef8bc9209ffcfaa03be096f4bf113daad82a92720da126ebef612c`.
Clean quote SHA-256: `00eb42ca5d60986d434e1fdc9ccde281a2ebb17b9d1997bceb60aa7420d511e9`.

Nothing was pushed to GitHub or uploaded to Kaggle. The previous power-sector
dataset, inverse-calibration trainers and frozen V2 outputs were not replaced.
