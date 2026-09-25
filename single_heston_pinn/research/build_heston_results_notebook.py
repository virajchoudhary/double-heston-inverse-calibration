from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Heston_Double_Heston_Results.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# Single Heston and Double Heston — validated NSE results

This notebook displays the locked Single-Heston and Double-Heston results for the NSE power-sector option dataset. It **does not invent, interpolate, or replace observations**. It reads the audited result artifacts already produced by the project, verifies the locked source hash, recalculates headline metrics from row-level predictions, and renders comparison graphs.

Important interpretation:

- Parameter selection used training/validation data and was frozen before the historical test calculation.
- The identical comparison contains 8,814 rows. Four additional Double-Heston rows produced invalid implied volatilities; they are disclosed separately and are not imputed.
- The historical test range was viewed during earlier model development, so it is not pristine final-holdout evidence.
- Scoring is conditional on the realized target-session spot and listed strike/expiry coordinates. This is not a pure pre-market forecast of future spot or future contracts.
- A dashed 45° line in an actual-versus-forecast chart is only the ideal-reference line; it is not a fitted model line.
"""
    ),
    code(
        r"""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")
plt.style.use("seaborn-v0_8-whitegrid")

# The notebook is designed to run from the project root in VS Code.
ROOT = Path.cwd().resolve()
if not (ROOT / "outputs" / "heston_handoff").exists():
    for candidate in [ROOT.parent, Path.home() / "Documents" / "Options pricing"]:
        if (candidate / "outputs" / "heston_handoff").exists():
            ROOT = candidate.resolve()
            break
if not (ROOT / "outputs" / "heston_handoff").exists():
    raise FileNotFoundError("Run this notebook from the 'Options pricing' project directory.")

LOCKED_INPUT_SHA256 = "a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3"
PATHS = {
    "model_input": ROOT / "outputs/019fc8a0/model_input_option_prices.csv",
    "audit_summary": ROOT / "outputs/heston_handoff/independent_heston_audit_summary.json",
    "audit_checks": ROOT / "outputs/heston_handoff/independent_heston_audit_checks.csv",
    "single_parameters": ROOT / "outputs/heston_handoff/single_heston_selected_parameter_catalog.csv",
    "double_parameters": ROOT / "outputs/heston_handoff/double_heston_selected_parameter_catalog.csv",
    "comparison_summary": ROOT / "outputs/single_double_heston_comparison/single_double_summary.json",
    "metrics": ROOT / "outputs/single_double_heston_comparison/single_double_test_metrics.csv",
    "predictions": ROOT / "outputs/single_double_heston_comparison/single_double_identical_test_predictions.csv",
    "invalid_double_rows": ROOT / "outputs/single_double_heston_comparison/double_heston_test_prediction_failures.csv",
    "integrity": ROOT / "outputs/single_double_heston_comparison/single_double_integrity_checks.csv",
    "single_controlled": ROOT / "outputs/single_heston/single_heston_controlled_tests.json",
    "double_controlled": ROOT / "outputs/single_double_heston_comparison/double_heston_controlled_tests.json",
    "postfit_summary": ROOT / "outputs/double_heston_stability_optimized/optimized_double_heston_summary.json",
    "postfit_rows": ROOT / "outputs/double_heston_stability_optimized/single_old_double_optimized_double_identical_rows.csv",
    "postfit_integrity": ROOT / "outputs/double_heston_stability_optimized/optimized_double_heston_integrity_checks.csv",
    "train_cv_summary": ROOT / "outputs/double_heston_train_cv_optimized/train_cv_optimized_double_heston_summary.json",
    "train_cv_rows": ROOT / "outputs/double_heston_train_cv_optimized/single_old_double_train_cv_double_identical_rows.csv",
    "train_cv_integrity": ROOT / "outputs/double_heston_train_cv_optimized/train_cv_double_heston_integrity_checks.csv",
    "state_scale_summary": ROOT / "outputs/double_heston_state_scale_audit/state_scale_audit_summary.json",
    "state_scale_rows": ROOT / "outputs/double_heston_state_scale_audit/state_scaled_double_heston_test_predictions.csv",
    "state_scale_integrity": ROOT / "outputs/double_heston_state_scale_audit/state_scale_integrity_checks.csv",
}

missing = [str(path) for path in PATHS.values() if not path.exists()]
if missing:
    raise FileNotFoundError("Missing required project artifacts:\n" + "\n".join(missing))

print(f"Project root: {ROOT}")
print(f"Python: {platform.python_version()} | pandas: {pd.__version__} | NumPy: {np.__version__}")
"""
    ),
    markdown(
        r"""
## 1. Authenticity and independent integrity checks

The source file is hashed byte-for-byte before any result is displayed. The audit evidence is loaded from the independent handoff audit, not reconstructed from narrative claims.
"""
    ),
    code(
        r"""
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def passed(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


observed_sha256 = sha256_file(PATHS["model_input"])
audit_summary = json.loads(PATHS["audit_summary"].read_text())
comparison_summary = json.loads(PATHS["comparison_summary"].read_text())
audit_checks = pd.read_csv(PATHS["audit_checks"])
integrity_checks = pd.read_csv(PATHS["integrity"])

assert observed_sha256 == LOCKED_INPUT_SHA256, "Locked NSE model-input hash mismatch"
assert observed_sha256 == audit_summary["input_sha256"] == comparison_summary["input_sha256"]
assert len(audit_checks) == audit_summary["checks_total"] == 60
assert passed(audit_checks["passed"]).all()
assert passed(integrity_checks["passed"]).all()
assert audit_summary["critical_failures"] == 0

authenticity = pd.DataFrame(
    [
        ["Locked model-input SHA-256", observed_sha256],
        ["Independent checks passed", f"{audit_summary['checks_passed']} / {audit_summary['checks_total']}"],
        ["Critical failures", audit_summary["critical_failures"]],
        ["Warnings/limitations recorded", audit_summary["warnings_recorded"]],
        ["Audit verdict", audit_summary["verdict"]],
        ["Identical finite comparison rows", comparison_summary["common_finite_test_rows"]],
        ["Documented invalid Double-Heston rows", comparison_summary["double_heston_invalid_test_rows"]],
    ],
    columns=["Evidence", "Observed value"],
)
display(authenticity)
"""
    ),
    markdown(
        r"""
## 2. Selected parameters

These are the per-stock parameters selected using training/validation evidence. For Double Heston, the ten displayed values comprise five slow-factor and five fast-factor values, including each factor's training-median initial variance. Boundary proximity is disclosed later as an identifiability warning.
"""
    ),
    code(
        r"""
single_params = pd.read_csv(PATHS["single_parameters"])
double_params = pd.read_csv(PATHS["double_parameters"])

single_columns = [
    "symbol", "selected_candidate", "kappa", "theta", "sigma", "rho",
    "v0_training_median", "validation_iv_rmse", "training_last_date",
]
display(Markdown("### Single Heston — 5 parameters per stock"))
display(single_params[single_columns].sort_values("symbol").reset_index(drop=True))

double_slow = [
    "symbol", "selected_candidate", "kappa_slow", "theta_slow", "sigma_slow",
    "rho_slow", "v0_slow_training_median", "training_last_date",
]
double_fast = [
    "symbol", "selected_candidate", "kappa_fast", "theta_fast", "sigma_fast",
    "rho_fast", "v0_fast_training_median", "training_last_date",
]
display(Markdown("### Double Heston — slow factor (5 of 10 parameters)"))
display(double_params[double_slow].sort_values("symbol").reset_index(drop=True))
display(Markdown("### Double Heston — fast factor (5 of 10 parameters)"))
display(double_params[double_fast].sort_values("symbol").reset_index(drop=True))
"""
    ),
    markdown(
        r"""
## 3. Recalculate the historical comparison

All headline metrics below are recalculated from the 8,814 identical finite option rows. No model receives extra rows in this comparison.
"""
    ),
    code(
        r"""
date_columns = ["trade_date", "expiry_date", "origin_date", "target_date"]
pred = pd.read_csv(PATHS["predictions"], parse_dates=date_columns)
saved_metrics = pd.read_csv(PATHS["metrics"])

required = {"row_key", "symbol", "market_iv", "single_heston_iv", "double_heston_iv", "target_date"}
assert required.issubset(pred.columns)
assert len(pred) == comparison_summary["common_finite_test_rows"] == 8_814
assert pred["row_key"].is_unique
assert np.isfinite(pred[["market_iv", "single_heston_iv", "double_heston_iv"]].to_numpy()).all()


def score(actual: pd.Series, forecast: pd.Series) -> dict:
    actual_array = actual.to_numpy(dtype=float)
    forecast_array = forecast.to_numpy(dtype=float)
    error = forecast_array - actual_array
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    denominator = float(np.sum((actual_array - actual_array.mean()) ** 2))
    r2 = float(1 - np.sum(error**2) / denominator)
    slope = float(np.polyfit(actual_array, forecast_array, 1)[0])
    return {"rows": len(actual_array), "IV RMSE": rmse, "IV MAE": mae, "IV bias": bias, "IV R²": r2, "forecast/actual slope": slope}


recalculated = pd.DataFrame(
    {
        "Single Heston": score(pred["market_iv"], pred["single_heston_iv"]),
        "Double Heston": score(pred["market_iv"], pred["double_heston_iv"]),
    }
).T

saved_overall = saved_metrics[saved_metrics["scope"].eq("ALL")].set_index("model")
for model in ["Single Heston", "Double Heston"]:
    assert np.isclose(recalculated.loc[model, "IV RMSE"], saved_overall.loc[model, "iv_rmse"], atol=1e-12)
    assert np.isclose(recalculated.loc[model, "IV R²"], saved_overall.loc[model, "iv_r2"], atol=1e-12)

display(recalculated)
print(f"Historical winner on identical rows: {comparison_summary['historical_model_preference']}")
bootstrap = comparison_summary["cluster_bootstrap_double_minus_single"]
print(
    "Cluster-bootstrap 95% interval for Double minus Single RMSE: "
    f"[{bootstrap['double_minus_single_rmse_low']:.6f}, {bootstrap['high']:.6f}]"
)
"""
    ),
    markdown(
        r"""
## 4. Honest Double-Heston parameter-adjustment search

Three additional adjustments were evaluated without allowing their later-period labels to select parameters: post-fit stability regularization, structural refitting with bounds selected inside the original training partition, and a validation-selected multiplier on the two forecast variance states. The same finite row intersection is used below for a direct comparison.
"""
    ),
    code(
        r"""
postfit_summary = json.loads(PATHS["postfit_summary"].read_text())
train_cv_summary = json.loads(PATHS["train_cv_summary"].read_text())
state_scale_summary = json.loads(PATHS["state_scale_summary"].read_text())

for check_path in [PATHS["postfit_integrity"], PATHS["train_cv_integrity"], PATHS["state_scale_integrity"]]:
    additional_checks = pd.read_csv(check_path)
    assert passed(additional_checks["passed"]).all(), f"Failed check in {check_path}"

postfit_rows = pd.read_csv(PATHS["postfit_rows"], usecols=["row_key", "optimized_double_iv"]).rename(
    columns={"optimized_double_iv": "postfit_stability_iv"}
)
train_cv_rows = pd.read_csv(PATHS["train_cv_rows"], usecols=["row_key", "optimized_double_iv"]).rename(
    columns={"optimized_double_iv": "train_cv_interior_iv"}
)
state_scale_rows = pd.read_csv(PATHS["state_scale_rows"], usecols=["row_key", "scaled_double_iv"])

search_common = (
    pred[["row_key", "market_iv", "single_heston_iv", "double_heston_iv"]]
    .merge(postfit_rows, on="row_key", validate="one_to_one")
    .merge(train_cv_rows, on="row_key", validate="one_to_one")
    .merge(state_scale_rows, on="row_key", validate="one_to_one")
    .dropna()
)
assert search_common.row_key.is_unique

search_models = {
    "Single Heston": "single_heston_iv",
    "Retained Double Heston": "double_heston_iv",
    "Post-fit stability candidate": "postfit_stability_iv",
    "Train-CV interior candidate": "train_cv_interior_iv",
    "State-scale candidate": "scaled_double_iv",
}
search_scores = pd.DataFrame(
    {label: score(search_common.market_iv, search_common[column]) for label, column in search_models.items()}
).T.sort_values("IV RMSE")
display(search_scores)

axis = search_scores["IV RMSE"].sort_values().plot.barh(
    figsize=(10, 4.8), color=["#1769aa", "#d1495b", "#9467bd", "#8c564b", "#7f7f7f"]
)
axis.set_title(f"Parameter-adjustment audit on {len(search_common):,} identical finite rows")
axis.set_xlabel("IV RMSE (lower is better)")
axis.set_ylabel("")
plt.tight_layout()
plt.show()

retained_rmse = search_scores.loc["Retained Double Heston", "IV RMSE"]
postfit_rmse = search_scores.loc["Post-fit stability candidate", "IV RMSE"]
train_cv_rmse = search_scores.loc["Train-CV interior candidate", "IV RMSE"]
display(
    Markdown(
        f"**Interpretation.** No Double-Heston adjustment improved on the retained validation-selected parameters: RMSE is {retained_rmse:.4f}, versus {postfit_rmse:.4f} after post-fit stability control and {train_cv_rmse:.4f} after train-only interior refitting. Validation also selected state multiplier 1.0 for every stock, so no state adjustment was accepted."
    )
)

adjustment_audit = pd.DataFrame(
    [
        ["Post-fit stability", postfit_summary["selected_recipe"], postfit_summary["selected_state_mode"], postfit_summary["selected_boundary_near_stocks"], "Rejected: higher RMSE"],
        ["Train-only interior refit", "per-stock bounds", "free two-factor state", train_cv_summary["selected_boundary_near_stocks"], "Rejected: higher RMSE"],
        ["Validation state scaling", "retained structural parameters", "scale = 1.0 for all stocks", None, "No adjustment selected"],
    ],
    columns=["Experiment", "Structural choice", "State choice", "Boundary-near stocks", "Decision"],
)
display(adjustment_audit)
"""
    ),
    markdown(
        r"""
## 5. Actual implied volatility versus both models

The faint points are individual option observations. The solid curve is a 20-quantile binned mean, which makes calibration/generalization shape easier to see. The dashed 45° line is the ideal `forecast = actual` reference and therefore has the same angle in both panels by definition.
"""
    ),
    code(
        r"""
models = [
    ("Single Heston", "single_heston_iv", "#1769aa"),
    ("Double Heston", "double_heston_iv", "#d1495b"),
]
limits = [
    float(pred[["market_iv", "single_heston_iv", "double_heston_iv"]].min().min()),
    float(pred[["market_iv", "single_heston_iv", "double_heston_iv"]].max().max()),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
for axis, (label, column, color) in zip(axes, models):
    axis.scatter(pred["market_iv"], pred[column], s=10, alpha=0.16, color=color, edgecolors="none", label="individual options")
    bins = pd.qcut(pred["market_iv"], q=20, duplicates="drop")
    curve = pred.assign(actual_bin=bins).groupby("actual_bin", observed=True)[["market_iv", column]].mean()
    axis.plot(curve["market_iv"], curve[column], color=color, marker="o", linewidth=2.2, markersize=4, label="20-bin mean")
    axis.plot(limits, limits, "--", color="#333333", linewidth=1.2, label="ideal reference")
    metrics_row = recalculated.loc[label]
    axis.set_title(f"{label}\nRMSE={metrics_row['IV RMSE']:.4f}, R²={metrics_row['IV R²']:.3f}")
    axis.set_xlabel("Actual market implied volatility")
    axis.set_ylabel("Model implied volatility")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.legend(loc="upper left")

fig.suptitle("Identical-row NSE comparison: actual IV vs model IV", fontsize=15, fontweight="bold")
fig.tight_layout()
plt.show()

display(
    Markdown(
        f'''**Interpretation.** Single Heston is closer overall (RMSE {recalculated.loc['Single Heston', 'IV RMSE']:.4f} versus {recalculated.loc['Double Heston', 'IV RMSE']:.4f}). Both models compress the volatility range: their forecast/actual slopes are {recalculated.loc['Single Heston', 'forecast/actual slope']:.3f} and {recalculated.loc['Double Heston', 'forecast/actual slope']:.3f}, below the ideal value of 1.'''
    )
)
"""
    ),
    markdown(
        r"""
## 6. Stock-level accuracy and a chronological example

The grouped bars compare out-of-sample historical-test RMSE on each stock's identical rows. The chronological chart uses daily cross-sectional mean IV for one editable stock, keeping the Single and Double paths visually separate.
"""
    ),
    code(
        r"""
stock_metrics = saved_metrics[~saved_metrics["scope"].eq("ALL")].copy()
stock_rmse = stock_metrics.pivot(index="scope", columns="model", values="iv_rmse")
stock_rmse["Double minus Single"] = stock_rmse["Double Heston"] - stock_rmse["Single Heston"]
stock_rmse["Winner"] = np.where(stock_rmse["Double minus Single"] < 0, "Double Heston", "Single Heston")
stock_rmse = stock_rmse.sort_values("Single Heston")

axis = stock_rmse[["Single Heston", "Double Heston"]].plot(
    kind="barh", figsize=(11, 6.5), color=["#1769aa", "#d1495b"], width=0.78
)
axis.set_title("Historical-test IV RMSE by stock — identical finite rows")
axis.set_xlabel("IV RMSE (lower is better)")
axis.set_ylabel("Stock")
axis.legend(loc="lower right")
plt.tight_layout()
plt.show()

single_stock_wins = int((stock_rmse["Winner"] == "Single Heston").sum())
double_stock_wins = int((stock_rmse["Winner"] == "Double Heston").sum())
display(Markdown(f"**Interpretation.** Single Heston has lower RMSE for {single_stock_wins} of {len(stock_rmse)} stocks; Double Heston wins only {double_stock_wins}. Shorter bars mean better historical-test accuracy."))

display(stock_rmse)
print("Stocks where Double Heston has lower RMSE:", ", ".join(comparison_summary["double_better_test_rmse_symbols"]))
"""
    ),
    code(
        r"""
# Change this symbol and rerun the cell in VS Code to inspect another stock.
SYMBOL_TO_PLOT = "POWERGRID"
available_symbols = sorted(pred["symbol"].unique())
if SYMBOL_TO_PLOT not in available_symbols:
    raise ValueError(f"Choose one of: {available_symbols}")

daily = (
    pred[pred["symbol"].eq(SYMBOL_TO_PLOT)]
    .groupby("target_date", as_index=False)[["market_iv", "single_heston_iv", "double_heston_iv"]]
    .mean()
)

fig, axis = plt.subplots(figsize=(13, 5))
axis.plot(daily["target_date"], daily["market_iv"], color="#111111", linewidth=2.4, marker="o", markersize=4, label="Actual market IV")
axis.plot(daily["target_date"], daily["single_heston_iv"], color="#1769aa", linewidth=1.8, marker="s", markersize=3, label="Single Heston")
axis.plot(daily["target_date"], daily["double_heston_iv"], color="#d1495b", linewidth=1.8, marker="^", markersize=3, label="Double Heston")
axis.set_title(f"{SYMBOL_TO_PLOT}: daily cross-sectional mean IV")
axis.set_xlabel("Target session")
axis.set_ylabel("Mean implied volatility")
axis.legend(ncol=3)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

daily_single_rmse = float(np.sqrt(np.mean((daily["single_heston_iv"] - daily["market_iv"]) ** 2)))
daily_double_rmse = float(np.sqrt(np.mean((daily["double_heston_iv"] - daily["market_iv"]) ** 2)))
daily_winner = "Single Heston" if daily_single_rmse < daily_double_rmse else "Double Heston"
display(Markdown(f"**Interpretation.** For {SYMBOL_TO_PLOT}, {daily_winner} follows the daily mean market IV more closely (daily-mean RMSE {min(daily_single_rmse, daily_double_rmse):.4f} versus {max(daily_single_rmse, daily_double_rmse):.4f}). This averaged view shows session-level tracking, not individual-option accuracy."))
"""
    ),
    markdown(
        r"""
## 7. Invalid-row disclosure and coverage

Invalid Double-Heston values are not replaced with the Single-Heston value, a baseline, zero, or a fabricated estimate. They are excluded from the identical finite-row score and retained in a separate failure file.
"""
    ),
    code(
        r"""
invalid_rows = pd.read_csv(PATHS["invalid_double_rows"])
assert len(invalid_rows) == comparison_summary["double_heston_invalid_test_rows"] == 4
assert invalid_rows["row_key"].is_unique
assert invalid_rows["double_heston_iv"].isna().all()
assert set(invalid_rows["row_key"]).isdisjoint(set(pred["row_key"]))

coverage = pd.DataFrame(
    {
        "Measure": ["Source-universe test rows", "Identical finite rows", "Invalid Double-Heston rows", "Finite common coverage"],
        "Value": [
            comparison_summary["test_source_universe_rows"],
            comparison_summary["common_finite_test_rows"],
            comparison_summary["double_heston_invalid_test_rows"],
            comparison_summary["finite_common_test_coverage"],
        ],
    }
)
display(coverage)
display(invalid_rows[["symbol", "trade_date", "expiry_date", "option_type", "strike", "market_iv", "double_heston_price", "double_heston_iv", "row_key"]])
"""
    ),
    markdown(
        r"""
## 8. Controlled synthetic recovery tests

Synthetic tests are separate from NSE results. They check implementation behavior against known parameters, multiple starting values, and 1% price noise. For Double Heston, the controlled recovery file checks the eight structural parameters; the two time-varying initial variances are state estimates, while all ten selected per-stock values are shown above.
"""
    ),
    code(
        r"""
single_controlled = json.loads(PATHS["single_controlled"].read_text())
double_controlled = json.loads(PATHS["double_controlled"].read_text())
assert single_controlled["all_checks_passed"] is True
assert double_controlled["all_checks_passed"] is True
assert single_controlled["synthetic_data_is_separate_from_nse_results"] is True
assert double_controlled["synthetic_data_is_separate_from_nse_results"] is True

single_names = ["kappa", "theta", "sigma", "rho", "v0"]
single_recovery = pd.DataFrame(
    {
        "parameter": single_names,
        "known": single_controlled["known_parameters"],
        "exact recovery": single_controlled["exact_recovered_parameters"],
        "1% noise recovery": single_controlled["noisy_recovered_parameters"],
    }
)
double_names = ["kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "kappa_fast", "theta_fast", "sigma_fast", "rho_fast"]
double_recovery = pd.DataFrame(
    {
        "parameter": double_names,
        "known": double_controlled["known_structural_parameters"],
        "exact recovery": double_controlled["exact_recovered_structural_parameters"],
        "1% noise recovery": double_controlled["noisy_recovered_structural_parameters"],
    }
)

display(Markdown("### Single-Heston controlled recovery"))
display(single_recovery)
display(Markdown("### Double-Heston controlled structural recovery"))
display(double_recovery)

controlled_summary = pd.DataFrame(
    [
        ["Single Heston", single_controlled["exact_price_rmse"], single_controlled["exact_max_relative_parameter_error"], single_controlled["noisy_price_rmse"]],
        ["Double Heston", double_controlled["exact_price_rmse"], double_controlled["exact_max_relative_structural_error"], double_controlled["noisy_price_rmse"]],
    ],
    columns=["model", "exact price RMSE", "exact max relative parameter error", "1% noise price RMSE"],
).set_index("model")
display(controlled_summary)
"""
    ),
    markdown(
        r"""
## 9. Audit warnings and honest conclusion

A passing warning check means the limitation was successfully detected and disclosed; it does not turn the limitation into a strength.
"""
    ),
    code(
        r"""
audit_breakdown = (
    audit_checks.assign(passed_flag=passed(audit_checks["passed"]))
    .groupby(["category", "severity"], dropna=False)
    .agg(checks=("check", "size"), passed=("passed_flag", "sum"))
    .reset_index()
)
warnings = audit_checks[audit_checks["severity"].astype(str).str.lower().eq("warning")].copy()
display(audit_breakdown)
display(Markdown("### Recorded warnings and limitations"))
display(warnings[["category", "check", "observed", "detail"]].reset_index(drop=True))

display(
    Markdown(
        f'''
### Evidence-based conclusion

- **Historical preference:** {comparison_summary['historical_model_preference']}.
- **Single-Heston RMSE:** {comparison_summary['single_heston']['iv_rmse']:.6f}; **Double-Heston RMSE:** {comparison_summary['double_heston']['iv_rmse']:.6f} on the same {comparison_summary['common_finite_test_rows']:,} rows.
- The Double-Heston minus Single-Heston bootstrap interval is entirely positive, so this historical sample favors Single Heston.
- Three stability-focused Double-Heston adjustment searches failed to improve RMSE; the validation-selected state multiplier was 1.0 for every stock. The retained Double-Heston catalog above is therefore the least-bad honest Double-Heston choice tested, not a claimed global optimum.
- Every one of the {comparison_summary['candidate_boundary_flags']} Double-Heston candidates is boundary-near. Its parameters should not be described as uniquely identified or globally optimal.
- These are retrospective, conditional historical-test results. Future dates must be locked before arrival for pristine forward validation.
'''
    )
)
"""
    ),
    markdown(
        r"""
## 10. Optional reproducibility commands

The audited pricing and calibration implementations remain in the project `.py` files. Keeping one implementation avoids notebook/script drift. Leave `RERUN_MODELS = False` for instant display of the locked outputs. Set it to `True` only when you intentionally want to regenerate the artifacts; full calibration can take substantial time.
"""
    ),
    code(
        r"""
source_files = [
    ROOT / "single_heston.py",
    ROOT / "double_heston.py",
    ROOT / "forecast_single_heston_next_day.py",
    ROOT / "compare_single_double_heston.py",
    ROOT / "optimize_double_heston_stable.py",
    ROOT / "optimize_double_heston_train_cv.py",
    ROOT / "validate_double_heston_state_scaling.py",
    ROOT / "audit_heston_handoff.py",
]
source_manifest = pd.DataFrame(
    [{"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in source_files]
)
display(source_manifest)

RERUN_MODELS = False
if RERUN_MODELS:
    commands = [
        [sys.executable, "forecast_single_heston_next_day.py"],
        [sys.executable, "compare_single_double_heston.py", "--reuse-candidates"],
        [sys.executable, "audit_heston_handoff.py"],
    ]
    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
else:
    print("Locked artifacts displayed. Set RERUN_MODELS = True to regenerate them intentionally.")
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Options pricing (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
