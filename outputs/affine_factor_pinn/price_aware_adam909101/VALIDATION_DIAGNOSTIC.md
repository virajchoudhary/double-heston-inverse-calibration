# Post-selection validation diagnostic

The selected step-3500 weights reproduced the recorded reserved weighted-price
RMSE exactly: **0.0012931310965554567**. This score is not exact IV RMSE.

When the actual Black inverse was applied to all **16,128 reserved validation
quotes**, **12 predictions were noninvertible** under the existing inversion
validity checks. Whole-set actual IV RMSE is therefore **undefined**, not an
RMSE computed after silently dropping those quotes. No prices were clipped,
no references changed, and no alternate checkpoint was retrospectively selected.

This exposes a limitation of selecting by target-vega-weighted price error:
lower surrogate loss need not produce an admissible IV at every quote. The
model is not validated for deployment. The separately run exposed12 assessment
retains its original invalid-price and recovery failure rules.

Reproduction: load the selected float64 `model.pt`, use `load_surfaces` from
`scripts/mentor_dh_pinn/finetune_factor_prices.py` on the archived corpus, and
evaluate indices 896 through 1023 with `neural_call_prices`. Apply
`sqrt(invert_total_variance(price, x) / tau)` to every prediction, count
nonfinite values before calculating any aggregate IV error. The loss target is
the archived synthetic IV, not an exposed12 observation. This is a post-fit
diagnostic and never supplies unknown calibration parameters.
