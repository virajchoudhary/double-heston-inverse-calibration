# Heston and Double-Heston Teammate Context

## Document status

- Version date: 06 August 2026
- Source: `handoff/Heston_Double_Heston_Validated_Teammate_Handoff_FINAL.pdf`
- Source scope: Single Heston and Double Heston only
- Audit verdict: `VALIDATED_WITH_DISCLOSED_LIMITATIONS`
- Independent audit result: 60/60 checks passed; five existing automated test modules exited successfully
- Locked model-input SHA-256: `a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3`

This Markdown file is a technical companion to, not a replacement for, the validated PDF. It preserves the handoff's principal equations, numerical findings, limitations, and continuation rules. The existing work uses power-sector stock options. The final approved capstone ANN work must move to NIFTY end-of-day European option surfaces.

## Audit verdict and honesty boundary

The handoff keeps the locked Single Heston model as the historical winner and does not promote the current Double Heston model. This is a reproducibility-qualified historical conclusion, not evidence of guaranteed future performance. The historical test graph was viewed during earlier development. Next-session surface scores use realized target spot and the target session's listed strike/expiry coordinates. Future locked NSE dates are required for pristine forward confirmation.

No row-level evidence showed test labels entering candidate fitting or selection: candidates were trained on train data and selected with validation data. That does not prove zero overfitting or that researchers never viewed aggregate test results.

## Locked dataset details

- Model input: `outputs/019fc8a0/model_input_option_prices.csv`
- Clean release rows: 572,512
- Model-ready rows: 215,636
- Model-ready stocks: 11 power-sector stocks
- Original row key: symbol, trade date, exact expiry, option type, strike
- Original source: `raw/nse_fo_bhavcopies/<year>/...csv.zip`
- Missing referenced raw files: 0
- Single-Heston test universe: 8,818 unique rows
- Double-Heston finite comparison coverage: 8,814 rows, or 99.9546%
- Double-Heston failures: four invalid implied-volatility rows, retained and not imputed

Authenticity in the PDF means evaluated rows resolve to local raw NSE FO bhavcopy paths and the locked input hash. It does not independently certify NSE infrastructure or re-download every bhavcopy.

## Data split and information boundary

Dates are split independently within each stock: 70% train, 15% validation, and 15% test. Every training date precedes the stock's first validation date. Validation selects candidates; test data score the already selected candidates. Validation and test row keys are disjoint. Each saved origin date strictly precedes its target date by one to four calendar days. Exact expiry dates and days to expiry are preserved, including historical Tuesday, Wednesday, and Thursday expiries.

The current next-session evaluation is a conditional volatility-surface forecast. It uses realized target spot and observed target contract coordinates, but not target option prices. Origin-state tables contain no target option-price or target-implied-volatility label.

## Single Heston definition

Under the risk-neutral measure, the implemented model is:

```text
dS(t) / S(t) = (r - q) dt + sqrt(v(t)) dW_S(t)
dv(t) = kappa [theta - v(t)] dt + sigma sqrt(v(t)) dW_v(t)
d<W_S, W_v> = rho dt
```

The parameter meanings are:

- `kappa`: mean-reversion speed
- `theta`: positive long-run variance
- `sigma`: volatility of variance
- `rho`: spot/variance correlation, historically constrained inside `(-0.98, 0.98)`
- `v0`: origin-date-specific variance state, propagated to the target

The strict Feller diagnostic is `2 * kappa * theta - sigma^2 > 0`. Pricing uses a Little-Heston-trap characteristic function with 64-point Gauss-Laguerre integration, put-call parity for puts, and Black-Scholes inversion for implied volatility. The training median of `v0` is descriptive and must not replace the live origin-specific state.

## Double Heston definition

The implemented Double Heston characteristic function is the product of two constrained Heston-factor characteristic functions, represented additively in log-characteristic exponents. Instantaneous variance is decomposed into slow and fast factors:

```text
dv_i(t) = kappa_i [theta_i - v_i(t)] dt
          + sigma_i sqrt(v_i(t)) dW_vi(t), i in {slow, fast}
total state = v_slow + v_fast
```

Each factor state propagates from an origin date across the exact calendar gap as:

```text
theta_i + [v_i(origin) - theta_i] * exp(-kappa_i * delta_calendar / 365)
```

Operationally the model has eight structural parameters and two surface-specific initial variance states.

## Locked ten-parameter order

The ordinary ANN uses exactly this output order:

```python
[
    "kappa_slow",
    "theta_slow",
    "sigma_slow",
    "rho_slow",
    "v0_slow",
    "kappa_fast",
    "theta_fast",
    "sigma_fast",
    "rho_fast",
    "v0_fast",
]
```

The ordering rule is `kappa_slow < kappa_fast`. Both factors must separately satisfy `2 * kappa_i * theta_i - sigma_i^2 > 0`. Each correlation lies strictly inside `(-1, 1)`, and the joint disk satisfies `rho_slow^2 + rho_fast^2 < 1`. All kappa, theta, sigma, and v0 parameters are strictly positive.

## Historical results

On the locked original next-session evaluation, Single Heston used 8,818 rows and achieved IV RMSE 0.044469, IV MAE 0.028048, bias -0.004604, and R2 0.814899. A prior-session median-IV baseline on the same 8,818 rows had RMSE 0.068079, MAE 0.042060, bias -0.031587, and R2 0.566172.

On 8,814 identical finite rows:

| Model | IV RMSE | IV MAE | Bias | R2 |
|---|---:|---:|---:|---:|
| Single Heston | 0.044200 | 0.027957 | -0.004502 | 0.816578 |
| Double Heston | 0.045510 | 0.028264 | -0.005601 | 0.805544 |

The stock-session bootstrap interval for Double minus Single RMSE was `[0.000607, 0.001993]`, entirely positive, favoring Single Heston. Double Heston won only for JSWENERGY (0.039108 versus 0.043534) and TORNTPOWER (0.050364 versus 0.056687); Single Heston won the other nine stocks.

All 55 Double-Heston candidates were boundary-near. The selected historical Double-Heston tables in the PDF are reproducibility artifacts, not uniquely identified economic truth and not supervised ANN labels.

A later validation-only guard improved the historical Single-Heston RMSE from 0.044469 to 0.036590 and slope from 0.806598 to 0.878963. It was designed after the historical test graph had been viewed, so it is explicitly retrospective and needs future locked NSE confirmation.

## Controlled synthetic recovery

The controlled recovery checks validate pricing/calibration mechanics, not the truth of Heston dynamics for real NSE data:

| Condition | Parameter recovery | Price RMSE |
|---|---:|---:|
| Single Heston, exact prices | max relative error 3.222e-10 | 2.817e-11 |
| Single Heston, 1% price noise | structural set remains close | 0.022329 |
| Double Heston, exact prices | max structural error 1.189e-05 | 3.679e-07 |
| Double Heston, 1% price noise | max structural error 28.875% | 0.026107 |

Small pricing error can coexist with materially displaced Double Heston parameters. Optimization convergence or repricing accuracy alone is not evidence of unique parameter recovery.

## Identifiability, leakage, and overfitting warnings

- Three-fold strike cross-fitting held out 4,709 unique quotes and produced Heston RMSE 0.030028.
- Within-surface centered R2 was 0.617874 versus pooled R2 0.903410; much pooled fit comes from same-day level calibration.
- A flat same-day anchor had RMSE 0.040577.
- A quadratic same-day smile had RMSE 0.014712 and beat Heston for same-day strike interpolation.
- Prior-available Heston state RMSE was 0.042245; shuffled-state RMSE was 0.068991.
- Single-Heston half-sample fits produced 39 boundary flags across 22 fits.
- Double Heston produced 55 boundary-near flags across 55 candidates.
- Similar option prices can be produced by materially different Double Heston parameter combinations.
- No finite historical experiment proves zero overfitting, a global optimum, or unique economics.
- The model universe is 11 model-ready power-sector stocks, not every power-sector security and not NIFTY.
- Filtered, liquid, parity-consistent options do not guarantee generalization to stale, illiquid, or rejected quotes.

## Continuation protocol

Freeze the teammate's `single_heston.py`, `double_heston.py`, `forecast_single_heston_next_day.py`, `compare_single_double_heston.py`, selected parameter catalogs, and checksum manifests. Do not use previously viewed historical test results to alter starts, bounds, filters, correction thresholds, or evaluation rows.

Acquire authentic future NSE sessions strictly after 03 August 2026, retain raw-source hashes and exact expiries, and record every exclusion reason. Build each origin state using origin-date information only and propagate over the exact calendar gap. Never fill or fabricate a failed model IV. Compare Single and Double on identical row keys, retain all failure rows, and bootstrap by stock-session rather than individual option rows.

The PDF recommends preregistering mean stock-session IV RMSE as the primary metric, a 95% stock-session cluster bootstrap interval for Double minus Single RMSE, at least 99.9% coverage, and a promotion rule requiring the interval to lie entirely below zero without gains being concentrated in one or two names. It also requires substantially fewer boundary-near fits and lower start/half-sample parameter ranges.

## Referenced teammate source-code files

The PDF references these project-relative files, but none were available in the current workspace at takeover time:

- `single_heston.py`
- `double_heston.py`
- `forecast_single_heston_next_day.py`
- `audit_single_heston_overfitting.py`
- `compare_single_double_heston.py`
- `audit_heston_handoff.py`
- `test_single_heston.py`
- `test_heston_overfit_audit.py`
- `test_heston_next_day_forecast.py`
- `test_heston_generalization.py`
- `test_double_heston.py`

The PDF also references locked model inputs, sanitized parameter catalogs, audit matrices, prediction failures, and comparison outputs beneath `outputs/`, but those artifacts were not supplied.

## Missing dependencies for ANN research development

The starter project can test its ANN infrastructure with an explicitly non-research dummy mapping. Genuine research remains blocked until the teammate supplies:

1. the frozen canonical `double_heston.py` and its required local dependencies;
2. an exact callable pricing contract or tests showing how to invoke it for vectorized calls and puts;
3. confirmed ten-parameter sampling bounds and their source/version;
4. Double Heston pricing and controlled synthetic-recovery tests;
5. any numerical integration settings required to reproduce the validated engine;
6. the final NIFTY surface-data contract, including rates, dividend treatment, quote filters, missing-grid/mask rules, and chronological split dates.

Historical calibrated parameters are not ANN ground truth. ANN supervision must use known parameter vectors sampled before pricing synthetic surfaces. No ANN or PINN research result exists in this starter project.
