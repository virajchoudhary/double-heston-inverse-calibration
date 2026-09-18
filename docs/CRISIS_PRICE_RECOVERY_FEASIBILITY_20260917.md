# Crisis-period price recovery: feasibility audit

**No crisis-model comparison has been run yet. Instrument approval is needed.**
The earlier study specifies ADANIPOWER. Its archived option data does not
support a convincing comparison on the requested principal crisis windows.
NIFTY index options look feasible, but using them would change the instrument
and leave the power-sector/individual-stock scope.

## What the archived NSE files actually contain

All available source ZIPs for these windows were checked against the retained
SHA-256 manifest before parsing. Legacy `CLOSE` and newer `ClsPric` are used,
not settlement prices. Exchange-reported expiry dates are retained; no weekday
is imposed. This is local provenance checking, not a new remote download audit.

| Event window | ADANIPOWER listed option rows | ADANIPOWER preliminary eligible dates | NIFTY preliminary eligible dates | NIFTY potential held-out strike pairs |
|---|---:|---:|---:|---:|
| March 2020 COVID crash | 6,678 | 2 | 21 | 691 |
| June 2025 Iran/US strikes | 0 | 0 | 21 | 1,667 |
| February 28-March 31, 2026 Iran conflict | 0 | 0 | 19 | 1,632 |
| April 2026 later conflict window | 3,130 | 10 | 20 | 1,516 |

These are **preliminary feasibility counts**, before full quote cleaning,
corporate-action checks, IV inversion, moneyness/vega filters and final
eligibility requirements. They are not final modeling sample sizes.

ADANIPOWER's two eligible March 2020 dates are March 4 and March 11, with only
four potential held-out strike pairs in total. They do not cover the later
March crash peak. April 2026 has more data, but substituting it for the
opening conflict period must be disclosed; it cannot be called the peak
volatility period without a separate model-independent volatility analysis.

The archive's ADANIPOWER gap agrees with NSE's
[March 9, 2026 circular](https://nsearchives.nseindia.com/content/circulars/FAOP73205.pdf),
which lists its F&O introduction effective April 1, 2026, subject to eligibility.
The [UN record of the US letter](https://digitallibrary.un.org/record/4105908)
dates commencement of US operations to February 28, 2026.
The [June 22, 2025 White House statement](https://www.whitehouse.gov/releases/2025/06/sunday-shows-president-trumps-pursuit-of-peace-through-strength-in-iran/)
documents the earlier US strikes. These sources define event dates; they do
not establish that a particular stock was at its maximum historical volatility.

## Preliminary gate, not a model-performance filter

- ADANIPOWER and NIFTY are examined separately; no pooling or silent substitution.
- Require positive traded closing prices and positive strikes; retain CE/PE pairs.
- Actual time to expiry is days/365, restricted to 7-100 days for this audit.
- Require six or more traded strike pairs per date/expiry. Reserve every
  third entire strike pair for potential testing before computing carry.
- Estimate forward and discount from the remaining call-put differences:
  `C-P = D*(F-K)`, using the existing study's six-step robust regression.
- Require positive finite forward, `exp(-.25*tau) <= D <= exp(.10*tau)`,
  parity residual RMSE/F <=.025, and anchor strike span/F >=.04.
- Legacy files have no published underlying spot field. No spot is invented:
  this preliminary gate normalizes by inferred forward instead of the old
  spot-based check, and does not infer separate dividend yield or spot.
- Entire input files pass hash checks; conflicting option keys hard-stop.
  No price fitting error or Single/Double Heston outcome enters eligibility.

Two tests pass: held-out quote corruption does not change anchor carry, and
an implausible discount is rejected rather than repaired. Other full dataset
checks remain to be performed before any modeling release.

## How an honest fixed-parameter comparison would work

Price reconstruction can be the objective even when true market parameters
are unknown. However, fixing all model parameters makes this a **forward price
test**, not inverse calibration. Network weights may learn the fixed model's
PDE solution; structural parameter values must remain frozen.

After instrument/window approval, declare both models' parameter sets before
examining their market errors. Use a comparable Single Heston baseline,
for example a disclosed moment-matched reduction of the fixed Double Heston
set, rather than an arbitrarily weaker single-factor volatility level.
Separate neural approximation error from model-to-market error by also
repricing each fixed parameter set with its independent reference engine.

Both models must use identical cleaned quotes, input conventions, date windows,
and computational budgets. Report all dates and paired date-level errors,
including losses/ties. Do not search crisis subwindows or parameter values
until Double Heston wins. An advantage in these windows would be conditional
evidence, not proof of general superiority or proof that two factors alone
caused the difference. Fixed literature parameters are not automatically
appropriate for NIFTY or ADANIPOWER.

## Files and pending choice

Audit runner: `scripts/mentor_dh_pinn/audit_crisis_option_coverage.py`.
Tests: `tests/test_crisis_option_coverage.py`.
Full source records and date/expiry groups:
`outputs/crisis_option_coverage_20260917/coverage_audit.json`.

Pending: permission to use **NIFTY 50 index options** for March 2020 and the
chosen Iran event window(s), instead of the previously specified ADANIPOWER.
No training, model replacement, database write, Kaggle upload or GitHub push
was performed in this feasibility step.
