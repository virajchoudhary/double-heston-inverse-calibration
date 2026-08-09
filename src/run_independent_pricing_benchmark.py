"""Run the frozen production-versus-independent-reference pricing benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CALL_OPTION, PARAMETER_NAMES
from .double_heston import price_double_heston_option
from .double_heston_reference import (
    REFERENCE_EPSABS,
    REFERENCE_EPSREL,
    REFERENCE_LIMIT,
    reference_double_heston_option,
)
from .utils import write_json

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "double_heston_benchmark_cases.json"
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs" / "double_heston_benchmark"
ABSOLUTE_TOLERANCE = 2e-5
RELATIVE_TOLERANCE = 2e-6
NO_ARBITRAGE_TOLERANCE = 1e-8
PARITY_TOLERANCE = 1e-10
NODE_COUNTS = (64, 96)

RESULT_COLUMNS = [
    "case_id", "pair_id", "case_category", "option_type", "spot", "strike", "maturity", "rate",
    "dividend_yield", "moneyness_definition", "maturity_bucket", "moneyness_bucket",
    *PARAMETER_NAMES, "reference_price", "reference_reliable", "reference_pass", "reference_failure",
    "reference_warning_count", "reference_p1_error", "reference_p2_error", "reference_runtime_seconds",
    "reference_no_arbitrage_pass", "reference_no_arbitrage_failure",
    "production_64_price", "production_64_runtime_seconds", "production_64_abs_difference",
    "production_64_relative_difference", "production_64_pass", "production_64_exception",
    "production_64_no_arbitrage_pass", "production_64_no_arbitrage_failure",
    "production_96_price", "production_96_runtime_seconds", "production_96_abs_difference",
    "production_96_relative_difference", "production_96_pass", "production_96_exception",
    "production_96_no_arbitrage_pass", "production_96_no_arbitrage_failure", "diagnostics_json",
]
FAILURE_COLUMNS = ["case_id", "node_count", "failure_type", "detail", "reference_reliable", "abs_difference", "tolerance"]


def _load_fixture() -> dict[str, Any]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if fixture["canonical_parameter_order"] != PARAMETER_NAMES:
        raise RuntimeError("benchmark fixture parameter order differs from canonical order")
    frozen = fixture["immutable_tolerances"]
    expected = {
        "absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE,
        "no_arbitrage": NO_ARBITRAGE_TOLERANCE, "parity": PARITY_TOLERANCE,
        "reference_epsabs": REFERENCE_EPSABS, "reference_epsrel": REFERENCE_EPSREL,
        "reference_limit": REFERENCE_LIMIT,
    }
    if frozen != expected:
        raise RuntimeError(f"benchmark fixture tolerances differ from immutable defaults: {frozen}")
    return fixture


def _maturity_bucket(maturity: float) -> str:
    return "short" if maturity <= 0.25 else "medium" if maturity <= 1.0 else "long"


def _moneyness_bucket(case: dict[str, Any]) -> tuple[str, str]:
    forward = case["spot"] * np.exp((case["rate"] - case["dividend_yield"]) * case["maturity"])
    ratio = case["strike"] / forward
    if case["option_type"] == CALL_OPTION:
        label = "ITM" if ratio < 0.95 else "ATM" if ratio <= 1.05 else "OTM"
        definition = "call: ITM K/F<0.95, ATM 0.95<=K/F<=1.05, OTM K/F>1.05"
    else:
        label = "ITM" if ratio > 1.05 else "ATM" if ratio >= 0.95 else "OTM"
        definition = "put: ITM K/F>1.05, ATM 0.95<=K/F<=1.05, OTM K/F<0.95"
    return label, definition


def _bounds(case: dict[str, Any], price: float) -> tuple[bool, str]:
    discounted_spot = case["spot"] * np.exp(-case["dividend_yield"] * case["maturity"])
    discounted_strike = case["strike"] * np.exp(-case["rate"] * case["maturity"])
    if case["option_type"] == CALL_OPTION:
        lower, upper = max(discounted_spot - discounted_strike, 0.0), discounted_spot
    else:
        lower, upper = max(discounted_strike - discounted_spot, 0.0), discounted_strike
    passed = bool(np.isfinite(price) and lower - NO_ARBITRAGE_TOLERANCE <= price <= upper + NO_ARBITRAGE_TOLERANCE)
    return passed, "" if passed else f"price={price}; lower={lower}; upper={upper}"


def _summary_for_node(rows: list[dict[str, Any]], node_count: int) -> dict[str, Any]:
    differences = np.asarray([row[f"production_{node_count}_abs_difference"] for row in rows], dtype=float)
    runtimes = np.asarray([row[f"production_{node_count}_runtime_seconds"] for row in rows], dtype=float)
    comparable = differences[np.isfinite(differences)]
    return {
        "rmse": float(np.sqrt(np.mean(comparable ** 2))) if comparable.size else None,
        "mae": float(np.mean(comparable)) if comparable.size else None,
        "maximum_absolute_difference": float(np.max(comparable)) if comparable.size else None,
        "passing_cases": int(sum(bool(row[f"production_{node_count}_pass"]) for row in rows)),
        "failing_cases": int(sum(not bool(row[f"production_{node_count}_pass"]) for row in rows)),
        "total_runtime_seconds": float(np.sum(runtimes)),
        "median_runtime_seconds": float(np.median(runtimes)),
        "p95_runtime_seconds": float(np.quantile(runtimes, 0.95)),
    }


def run_benchmark(output_directory: str | Path | None = None) -> dict[str, Any]:
    """Price every frozen case and write the seven benchmark artifacts."""
    fixture = _load_fixture()
    output = Path(output_directory) if output_directory is not None else OUTPUT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in sorted(fixture["cases"], key=lambda item: item["case_id"]):
        parameters = case["parameters"]
        moneyness_bucket, moneyness_definition = _moneyness_bucket(case)
        reference_start = time.perf_counter()
        try:
            reference_price, reference_diagnostics = reference_double_heston_option(
                case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
                case["option_type"], parameters,
            )
        except Exception as error:
            reference_price = float("nan")
            reference_diagnostics = {"reliable": False, "failure": f"{type(error).__name__}: {error}", "warnings": [], "p1": {}, "p2": {}}
        reference_runtime = time.perf_counter() - reference_start
        row: dict[str, Any] = {
            "case_id": case["case_id"], "pair_id": case["generation_metadata"]["pair_id"],
            "case_category": case["case_category"], "option_type": case["option_type"],
            "spot": case["spot"], "strike": case["strike"], "maturity": case["maturity"], "rate": case["rate"],
            "dividend_yield": case["dividend_yield"], "moneyness_definition": moneyness_definition,
            "maturity_bucket": _maturity_bucket(case["maturity"]), "moneyness_bucket": moneyness_bucket,
            **dict(zip(PARAMETER_NAMES, parameters, strict=True)), "reference_price": reference_price,
            "reference_reliable": bool(reference_diagnostics.get("reliable", False)),
            "reference_pass": False,
            "reference_failure": reference_diagnostics.get("failure") or "",
            "reference_warning_count": len(reference_diagnostics.get("warnings", [])),
            "reference_p1_error": reference_diagnostics.get("p1", {}).get("absolute_error_estimate", float("nan")),
            "reference_p2_error": reference_diagnostics.get("p2", {}).get("absolute_error_estimate", float("nan")),
            "reference_runtime_seconds": reference_runtime,
            "diagnostics_json": json.dumps(reference_diagnostics, sort_keys=True),
        }
        reference_no_arb, reference_no_arb_failure = _bounds(case, reference_price)
        row["reference_no_arbitrage_pass"], row["reference_no_arbitrage_failure"] = reference_no_arb, reference_no_arb_failure
        row["reference_pass"] = bool(row["reference_reliable"] and reference_no_arb)
        if not row["reference_reliable"]:
            failures.append({"case_id": case["case_id"], "node_count": "reference", "failure_type": "reference_unreliable",
                             "detail": row["reference_failure"] or "integration warning or error estimate exceeded tolerance",
                             "reference_reliable": False, "abs_difference": float("nan"), "tolerance": float("nan")})
        for node_count in NODE_COUNTS:
            key = f"production_{node_count}"
            started = time.perf_counter()
            try:
                production_price = price_double_heston_option(
                    case["spot"], case["strike"], case["maturity"], case["rate"], case["dividend_yield"],
                    case["option_type"], parameters, node_count=node_count,
                )
                exception = ""
            except Exception as error:
                production_price, exception = float("nan"), f"{type(error).__name__}: {error}"
            runtime = time.perf_counter() - started
            difference = abs(production_price - reference_price) if np.isfinite(production_price) and np.isfinite(reference_price) else float("inf")
            tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(reference_price) if np.isfinite(reference_price) else float("nan")
            production_no_arb, production_no_arb_failure = _bounds(case, production_price)
            passed = bool(row["reference_reliable"] and not exception and difference <= tolerance and production_no_arb)
            row.update({f"{key}_price": production_price, f"{key}_runtime_seconds": runtime,
                        f"{key}_abs_difference": difference, f"{key}_relative_difference": difference / abs(reference_price) if abs(reference_price) >= 1e-4 else float("nan"),
                        f"{key}_pass": passed, f"{key}_exception": exception,
                        f"{key}_no_arbitrage_pass": production_no_arb,
                        f"{key}_no_arbitrage_failure": production_no_arb_failure})
            if not passed:
                failures.append({"case_id": case["case_id"], "node_count": node_count,
                                 "failure_type": "production_exception" if exception else "tolerance_breach",
                                 "detail": exception or f"difference={difference}",
                                 "reference_reliable": row["reference_reliable"], "abs_difference": difference, "tolerance": tolerance})
            if not production_no_arb:
                failures.append({"case_id": case["case_id"], "node_count": node_count, "failure_type": "no_arbitrage",
                                 "detail": production_no_arb_failure, "reference_reliable": row["reference_reliable"],
                                 "abs_difference": float("nan"), "tolerance": NO_ARBITRAGE_TOLERANCE})
        if not reference_no_arb:
            failures.append({"case_id": case["case_id"], "node_count": "reference", "failure_type": "no_arbitrage",
                             "detail": reference_no_arb_failure, "reference_reliable": row["reference_reliable"],
                             "abs_difference": float("nan"), "tolerance": NO_ARBITRAGE_TOLERANCE})
        rows.append(row)
    import pandas as pd
    result_frame = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    maturity = result_frame.groupby("maturity_bucket", sort=True).agg(case_count=("case_id", "size"), reference_reliable=("reference_reliable", "all"),
        max_abs_difference_64=("production_64_abs_difference", "max"), max_abs_difference_96=("production_96_abs_difference", "max"),
        all_pass_64=("production_64_pass", "all"), all_pass_96=("production_96_pass", "all")).reset_index()
    maturity.to_csv(output / "benchmark_by_maturity.csv", index=False)
    moneyness = result_frame.groupby(["option_type", "moneyness_bucket"], sort=True).agg(case_count=("case_id", "size"), reference_reliable=("reference_reliable", "all"),
        max_abs_difference_64=("production_64_abs_difference", "max"), max_abs_difference_96=("production_96_abs_difference", "max"),
        all_pass_64=("production_64_pass", "all"), all_pass_96=("production_96_pass", "all")).reset_index()
    moneyness["definition"] = "strike/forward with option-direction-aware ITM/ATM/OTM definitions in benchmark_case_results.csv"
    moneyness.to_csv(output / "benchmark_by_moneyness.csv", index=False)
    quadrature = pd.DataFrame([{"comparison": "64_vs_reference", **_summary_for_node(rows, 64)}, {"comparison": "96_vs_reference", **_summary_for_node(rows, 96)},
        {"comparison": "64_vs_96", "rmse": float(np.sqrt(np.mean((result_frame["production_64_price"] - result_frame["production_96_price"]) ** 2))),
         "mae": float(np.mean(np.abs(result_frame["production_64_price"] - result_frame["production_96_price"]))),
         "maximum_absolute_difference": float(np.max(np.abs(result_frame["production_64_price"] - result_frame["production_96_price"])))}])
    quadrature.to_csv(output / "quadrature_comparison.csv", index=False)
    runtime = pd.DataFrame([{"method": "reference_scipy_quad", "total_seconds": float(result_frame["reference_runtime_seconds"].sum()), "median_seconds": float(result_frame["reference_runtime_seconds"].median()), "p95_seconds": float(result_frame["reference_runtime_seconds"].quantile(.95))},
        *[{"method": f"production_{n}_nodes", "total_seconds": float(result_frame[f"production_{n}_runtime_seconds"].sum()), "median_seconds": float(result_frame[f"production_{n}_runtime_seconds"].median()), "p95_seconds": float(result_frame[f"production_{n}_runtime_seconds"].quantile(.95))} for n in NODE_COUNTS]])
    runtime.to_csv(output / "runtime_comparison.csv", index=False)
    pair_errors: dict[str, dict[str, float]] = {}
    for pair_id, group in result_frame.groupby("pair_id", sort=True):
        calls, puts = group[group["option_type"] == "call"], group[group["option_type"] == "put"]
        if len(calls) == len(puts) == 1:
            call, put = calls.iloc[0], puts.iloc[0]
            parity = call["spot"] * np.exp(-call["dividend_yield"] * call["maturity"]) - call["strike"] * np.exp(-call["rate"] * call["maturity"])
            pair_errors[str(pair_id)] = {label: float(abs(call[f"{label}_price"] - put[f"{label}_price"] - parity)) for label in ("reference", "production_64", "production_96")}
    max_parity = {key: max((value[key] for value in pair_errors.values()), default=float("nan")) for key in ("reference", "production_64", "production_96")}
    parity_failures = {key: int(sum(value[key] > PARITY_TOLERANCE for value in pair_errors.values())) for key in max_parity}
    for pair_id, errors in pair_errors.items():
        for method, error in errors.items():
            if error > PARITY_TOLERANCE:
                failures.append({"case_id": pair_id, "node_count": method, "failure_type": "put_call_parity",
                                 "detail": f"paired parity error={error}", "reference_reliable": True,
                                 "abs_difference": error, "tolerance": PARITY_TOLERANCE})
    result_frame.to_csv(output / "benchmark_case_results.csv", index=False)
    pd.DataFrame(failures).reindex(columns=FAILURE_COLUMNS).to_csv(output / "benchmark_failures.csv", index=False)
    relative_values = np.concatenate([result_frame["production_64_relative_difference"].dropna().to_numpy(), result_frame["production_96_relative_difference"].dropna().to_numpy()])
    summary = {"schema_version": "1.0", "case_count": len(rows), "coverage": {"maturity_buckets": sorted(result_frame["maturity_bucket"].unique().tolist()), "moneyness_buckets": sorted(result_frame["moneyness_bucket"].unique().tolist()), "calls": int((result_frame["option_type"] == "call").sum()), "puts": int((result_frame["option_type"] == "put").sum())},
        "acceptance": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE, "non_negligible_reference_floor": 1e-4, "no_arbitrage": NO_ARBITRAGE_TOLERANCE, "parity": PARITY_TOLERANCE},
        "reference": {"integration_failure_or_unreliable_count": int((~result_frame["reference_reliable"]).sum()), "warning_count": int(result_frame["reference_warning_count"].sum()), "total_runtime_seconds": float(result_frame["reference_runtime_seconds"].sum()), "median_runtime_seconds": float(result_frame["reference_runtime_seconds"].median()), "p95_runtime_seconds": float(result_frame["reference_runtime_seconds"].quantile(.95))},
        "node_comparisons": {str(n): _summary_for_node(rows, n) for n in NODE_COUNTS}, "maximum_relative_difference_non_negligible": float(np.max(relative_values)) if relative_values.size else None,
        "parity": {"max_errors": max_parity, "failures": parity_failures},
        "no_arbitrage_failures": {"reference": int((~result_frame["reference_no_arbitrage_pass"]).sum()),
                                  "64": int((~result_frame["production_64_no_arbitrage_pass"]).sum()),
                                  "96": int((~result_frame["production_96_no_arbitrage_pass"]).sum())},
        "no_arbitrage_failures_total": int((~result_frame["reference_no_arbitrage_pass"]).sum()
                                          + (~result_frame["production_64_no_arbitrage_pass"]).sum()
                                          + (~result_frame["production_96_no_arbitrage_pass"]).sum()),
        "benchmark_pass": bool(not failures and result_frame["reference_pass"].all()
                               and result_frame["production_64_pass"].all()
                               and result_frame["production_96_pass"].all()
                               and not any(parity_failures.values())),
        "agreement_is_not_proof_of_correctness": True, "no_teammate_equivalence_claim": True}
    write_json(output / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
