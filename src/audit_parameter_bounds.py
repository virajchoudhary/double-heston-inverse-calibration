"""Audit provisional Double Heston bounds without changing their source file."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.stats import qmc

from .constants import CALL_OPTION, PARAMETER_NAMES, PUT_OPTION
from .double_heston import price_double_heston_surface
from .utils import write_json

ROOT = Path(__file__).resolve().parents[1]
PROVISIONAL_PATH = ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
REVIEWED_PATH = ROOT / "configs" / "parameter_sampling_REVIEWED.yaml"
OUTPUT = ROOT / "outputs" / "parameter_bounds_audit"
FREEZE = ROOT / "outputs" / "engine_freeze"
SEED = 20260806
RAW_CANDIDATE_COUNT = 5000
PRICE_LIMIT = 250
STRIKES = np.asarray([70., 80., 90., 100., 110., 120., 130.])
MATURITIES = np.asarray([.05, .25, .5, 1., 2., 5.])


def _load_bounds() -> dict[str, Any]:
    return yaml.safe_load(PROVISIONAL_PATH.read_text(encoding="utf-8"))


def _margin_row(candidate_id: int, vector: np.ndarray, hard: dict[str, Any]) -> dict[str, Any]:
    p = dict(zip(PARAMETER_NAMES, vector, strict=True))
    hard_distances = {name: min((p[name] - hard[name]["lower"]) / (hard[name]["upper"] - hard[name]["lower"]),
                                    (hard[name]["upper"] - p[name]) / (hard[name]["upper"] - hard[name]["lower"])) for name in PARAMETER_NAMES}
    slow_gap = 2 * p["kappa_slow"] * p["theta_slow"] - p["sigma_slow"] ** 2
    fast_gap = 2 * p["kappa_fast"] * p["theta_fast"] - p["sigma_fast"] ** 2
    slow_margin = slow_gap / (2 * p["kappa_slow"] * p["theta_slow"] + p["sigma_slow"] ** 2)
    fast_margin = fast_gap / (2 * p["kappa_fast"] * p["theta_fast"] + p["sigma_fast"] ** 2)
    correlation_disk = p["rho_slow"] ** 2 + p["rho_fast"] ** 2
    correlation_margin = 1.0 - correlation_disk
    ordering_margin = (p["kappa_fast"] - p["kappa_slow"]) / (hard["kappa_fast"]["upper"] - hard["kappa_slow"]["lower"])
    positivity = all(p[name] > 0 for name in ("kappa_slow", "theta_slow", "sigma_slow", "v0_slow", "kappa_fast", "theta_fast", "sigma_fast", "v0_fast"))
    hard_valid = all(hard[name]["lower"] <= p[name] <= hard[name]["upper"] for name in PARAMETER_NAMES)
    reasons: list[str] = []
    if not positivity: reasons.append("positivity")
    if not hard_valid: reasons.append("hard_numerical_safety_bounds")
    if not p["kappa_slow"] < p["kappa_fast"]: reasons.append("slow_fast_ordering")
    if not slow_gap > 0: reasons.append("slow_feller")
    if not fast_gap > 0: reasons.append("fast_feller")
    if not (-1 < p["rho_slow"] < 1 and -1 < p["rho_fast"] < 1): reasons.append("individual_correlation_bounds")
    if not correlation_disk < 1: reasons.append("joint_correlation_disk")
    minimum_constraint = min(slow_margin, fast_margin, correlation_margin, ordering_margin)
    minimum_hard = min(hard_distances.values())
    accepted = not reasons
    feller_near = bool(accepted and (0.0 <= slow_margin <= .05 or 0.0 <= fast_margin <= .05))
    correlation_disk_near = bool(accepted and 0.0 <= correlation_margin <= .05)
    weak_separation = bool(accepted and 0.0 <= ordering_margin <= .10)
    hard_bound_near = bool(accepted and 0.0 <= minimum_hard <= .05)
    boundary_near = bool(feller_near or correlation_disk_near or hard_bound_near)
    return {"candidate_id": candidate_id, **p, "positivity_valid": positivity, "hard_bounds_valid": hard_valid,
            "ordering_valid": p["kappa_slow"] < p["kappa_fast"], "slow_feller_gap": slow_gap, "fast_feller_gap": fast_gap,
            "slow_feller_margin": slow_margin, "fast_feller_margin": fast_margin, "correlation_disk_value": correlation_disk,
            "correlation_margin": correlation_margin, "ordering_margin": ordering_margin, "minimum_hard_bound_distance": minimum_hard,
            "minimum_normalized_constraint_distance": minimum_constraint, "constraint_violating": not accepted,
            "accepted_hard_bound_near": hard_bound_near, "accepted_feller_near": feller_near,
            "accepted_correlation_disk_near": correlation_disk_near, "accepted_weak_slow_fast_separation": weak_separation,
            "accepted_any_boundary_near": boundary_near, "boundary_near": boundary_near,
            "weak_slow_fast_separation": weak_separation, "correlation_disk_near": correlation_disk_near,
            "feller_near": feller_near, "accepted": accepted,
            "primary_rejection_reason": reasons[0] if reasons else "", "rejection_reasons": ";".join(reasons)}


def _select_priced(accepted: pd.DataFrame) -> pd.DataFrame:
    if len(accepted) <= PRICE_LIMIT: return accepted.sort_values("candidate_id").copy()
    ranked = accepted.sort_values(["minimum_normalized_constraint_distance", "candidate_id"], kind="stable")
    rng = np.random.default_rng(SEED)
    selected: list[pd.DataFrame] = []
    chunk_edges = np.linspace(0, len(ranked), 11, dtype=int)
    for start, stop in zip(chunk_edges[:-1], chunk_edges[1:], strict=True):
        stratum = ranked.iloc[start:stop]
        positions = np.sort(rng.choice(len(stratum), size=25, replace=False))
        selected.append(stratum.iloc[positions])
    return pd.concat(selected).sort_values("candidate_id").copy()


def _surface_metrics(vector: np.ndarray) -> dict[str, Any]:
    strikes = np.tile(STRIKES, len(MATURITIES) * 2)
    maturities = np.repeat(MATURITIES, len(STRIKES) * 2)
    option_types = np.tile(np.repeat([CALL_OPTION, PUT_OPTION], len(STRIKES)), len(MATURITIES))
    started = time.perf_counter()
    prices = price_double_heston_surface(100., strikes, maturities, .02, .01, option_types, vector, node_count=64)
    runtime = time.perf_counter() - started
    lower_spot = 100. * np.exp(-.01 * maturities)
    lower_strike = strikes * np.exp(-.02 * maturities)
    lower = np.where(option_types == CALL_OPTION, np.maximum(lower_spot - lower_strike, 0), np.maximum(lower_strike - lower_spot, 0))
    upper = np.where(option_types == CALL_OPTION, lower_spot, lower_strike)
    call_monotone = put_monotone = convex = True
    atm_calls: dict[str, float] = {}
    surface: list[float] = []
    for maturity in MATURITIES:
        mask = maturities == maturity
        call = prices[mask & (option_types == CALL_OPTION)]
        put = prices[mask & (option_types == PUT_OPTION)]
        call_monotone &= bool(np.all(np.diff(call) <= 1e-9))
        put_monotone &= bool(np.all(np.diff(put) >= -1e-9))
        convex &= bool(np.all(np.diff(call, n=2) >= -1e-8) and np.all(np.diff(put, n=2) >= -1e-8))
        atm_calls[f"{maturity:g}"] = float(call[3] / 100.)
        surface.extend((prices[mask] / 100.).tolist())
    atm = np.asarray(list(atm_calls.values()))
    return {"finite_price_rate": float(np.mean(np.isfinite(prices))), "no_arbitrage_valid": bool(np.all(prices >= lower - 1e-8) and np.all(prices <= upper + 1e-8)),
            "min_normalized_price": float(np.min(prices / 100.)), "max_normalized_price": float(np.max(prices / 100.)),
            "call_strike_monotonicity": call_monotone, "put_strike_monotonicity": put_monotone, "strike_convexity": convex,
            "atm_normalized_call_by_maturity": json.dumps(atm_calls, sort_keys=True), "term_structure_feature_variety": int(np.unique(np.round(atm, 8)).size),
            "calendar_monotonicity_not_imposed": True, "implied_volatility": "unavailable_skipped_no_validated_inversion", "runtime_seconds": runtime,
            "_surface": np.asarray(surface)}


def _write_reviewed_config(bounds: dict[str, Any], summary: dict[str, Any]) -> None:
    def ranged(source: str, name: str, entry: dict[str, Any], status: str = "REVIEW") -> dict[str, Any]:
        return {"lower": float(entry["lower"]), "upper": float(entry["upper"]), "source": source, "rationale": entry.get("rationale", "Retained for review from provisional evidence."),
                "status": status, "provisional": True, "reviewed": False}
    hard = bounds["hard_numerical_safety_bounds"]
    empirical = bounds["empirical_sampling_ranges"]["parameter_bounds"]
    proximity_metrics = summary["boundary_proximity"]["metrics"]
    accepted_near_any = proximity_metrics["accepted_near_any_boundary"]
    accepted_near_feller = proximity_metrics["accepted_near_either_feller"]
    payload = {
        "schema_version": "1.0", "status": "REVIEWED_EVIDENCE_NOT_FINANCIAL_APPROVAL",
        "seed": {"value": SEED, "source": "bounds audit specification", "rationale": "Reproducible Latin-hypercube sampling.", "status": "KEEP", "provisional": False, "reviewed": True},
        "method": {"value": "scipy.stats.qmc.LatinHypercube", "source": "bounds audit specification", "rationale": "Deterministic coverage of provisional empirical ranges.", "status": "KEEP", "provisional": False, "reviewed": True},
        "hard_numerical_validity_limits": {name: ranged("parameter_bounds_PROVISIONAL.yaml hard_numerical_safety_bounds", name, value, "KEEP") for name, value in hard.items()},
        "ann_synthetic_training_sampling_ranges": {name: ranged("parameter_bounds_PROVISIONAL.yaml empirical_sampling_ranges", name, value) for name, value in empirical.items()},
        "constraint_margins": {
            "slow_fast_ordering": {"value": 0.0, "source": "declared repository constraint", "rationale": "Strict validity remains required; margin adequacy is under review.", "status": "REVIEW", "provisional": True, "reviewed": False},
            "slow_feller": {"value": 0.0, "source": "declared repository constraint", "rationale": f"Strict validity remains required; accepted-valid Feller-near rate is {accepted_near_feller['proportion']:.4f}.", "status": "REVIEW", "provisional": True, "reviewed": False},
            "fast_feller": {"value": 0.0, "source": "declared repository constraint", "rationale": f"Strict validity remains required; accepted-valid Feller-near rate is {accepted_near_feller['proportion']:.4f}.", "status": "REVIEW", "provisional": True, "reviewed": False},
            "joint_correlation_disk": {"value": 0.0, "source": "declared repository constraint", "rationale": "Strict disk validity remains required.", "status": "REVIEW", "provisional": True, "reviewed": False}},
        "boundary_near_challenge_sampling": {"minimum_normalized_distance": {"value": .05, "source": "bounds audit specification", "rationale": f"Challenge-only accepted-valid boundary definition; observed rate {accepted_near_any['proportion']:.4f} ({accepted_near_any['count']}/{accepted_near_any['denominator']}).", "status": "REVIEW", "provisional": True, "reviewed": False}},
        "noise_test_sampling": {"relative_price_noise": {"lower": 0.0, "upper": .01, "source": "existing controlled validation", "rationale": "Controlled noise only, not a real-market claim.", "status": "REVIEW", "provisional": True, "reviewed": False}},
        "out_of_distribution_test_ranges": {name: ranged("provisional hard numerical bounds; OOD design unresolved", name, value) for name, value in hard.items()},
        "evidence": {
            "raw_candidate_count": {"value": RAW_CANDIDATE_COUNT, "source": "bounds audit output", "rationale": "Fixed audit population.", "status": "KEEP", "provisional": False, "reviewed": True},
            "accepted_count": {"value": summary["accepted_count"], "source": "bounds audit output", "rationale": "Observed after declared constraints.", "status": "REVIEW", "provisional": True, "reviewed": False},
            "priced_surface_count": {"value": summary["priced_surface_count"], "source": "bounds audit output", "rationale": "Capped audit-only production pricing sample.", "status": "KEEP", "provisional": False, "reviewed": True},
            "uniform_raw_bound_sampling_appropriate": {"value": summary["uniform_raw_bound_sampling_appropriate"], "source": "bounds audit output", "rationale": "Evidence-based design diagnostic only.", "status": "REVIEW", "provisional": True, "reviewed": False},
            "accepted_near_any_boundary": {"value": accepted_near_any["proportion"], "source": "bounds audit output accepted_valid_candidates", "rationale": f"Accepted-valid population only: {accepted_near_any['count']}/{accepted_near_any['denominator']} candidates are boundary-near.", "status": "REVIEW", "provisional": True, "reviewed": False},
            "accepted_near_either_feller": {"value": accepted_near_feller["proportion"], "source": "bounds audit output accepted_valid_candidates", "rationale": f"Accepted-valid population only: {accepted_near_feller['count']}/{accepted_near_feller['denominator']} candidates are Feller-near.", "status": "REVIEW", "provisional": True, "reviewed": False},
            "no_real_market_validation": {"value": True, "source": "milestone scope", "rationale": "Synthetic evidence is not market validation.", "status": "REQUIRE_FINANCIAL_REVIEW", "provisional": True, "reviewed": False},
            "no_teammate_equivalence_claim": {"value": True, "source": "milestone scope", "rationale": "Unavailable code was not compared.", "status": "KEEP", "provisional": False, "reviewed": True},
            "statement": {"value": "No real-market validation or teammate-equivalence evidence is claimed; financial review remains required.", "source": "milestone scope", "rationale": "Prevents unsupported readiness claims.", "status": "REQUIRE_FINANCIAL_REVIEW", "provisional": True, "reviewed": False}}}
    canonical_yaml = yaml.safe_dump(payload, sort_keys=True).replace("\r\n", "\n")
    REVIEWED_PATH.write_text(canonical_yaml, encoding="utf-8", newline="\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(summary: dict[str, Any], *, commands_passed: bool = False) -> dict[str, Any]:
    FREEZE.mkdir(parents=True, exist_ok=True)
    benchmark_path = ROOT / "outputs" / "double_heston_benchmark" / "benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else None
    write_json(FREEZE / "benchmark_summary.json", benchmark or {"missing": True})
    write_json(FREEZE / "parameter_sampling_summary.json", summary)
    sources = [ROOT / "src" / "double_heston.py", ROOT / "src" / "double_heston_reference.py", ROOT / "src" / "run_independent_pricing_benchmark.py",
               ROOT / "src" / "audit_parameter_bounds.py", PROVISIONAL_PATH, REVIEWED_PATH, ROOT / "tests" / "fixtures" / "double_heston_benchmark_cases.json"]
    checksums = {path.relative_to(ROOT).as_posix(): _sha(path) for path in sources if path.exists()}
    write_json(FREEZE / "source_checksums.json", checksums)
    write_json(FREEZE / "fixture_checksums.json", {ROOT.joinpath("tests/fixtures/double_heston_benchmark_cases.json").relative_to(ROOT).as_posix(): _sha(ROOT / "tests/fixtures/double_heston_benchmark_cases.json")})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema_version": "1.0", "timestamp_utc": datetime.now(UTC).isoformat(), "git_commit_before_new_commit": commit,
                "expected_baseline_commit": "dea6a19238e13fccf5243935ffb6df8199135595", "python_version": sys.version.split()[0],
                "numpy_version": np.__version__, "scipy_version": scipy.__version__, "production_node_counts": [64, 96],
                "reference_tolerances": {"epsabs": 1e-10, "epsrel": 1e-10, "limit": 500}, "canonical_parameter_order": PARAMETER_NAMES,
                "constraint_definitions": _load_bounds()["declared_constraints"], "no_teammate_equivalence_claim": True,
                "no_real_market_validation_claim": True, "ann_training_occurred": False}
    write_json(FREEZE / "engine_manifest.json", manifest)
    (FREEZE / "validation_commands.txt").write_text("python -m compileall .\npython -m pytest tests -q\npython -m src.run_independent_pricing_benchmark\npython -m src.audit_parameter_bounds\npython -m src.run_double_heston_validation\npython -m src.run_smoke_test\npython -m src.evaluate_repricing\n", encoding="utf-8")
    benchmark_blocked = benchmark is None or benchmark["reference"]["integration_failure_or_unreliable_count"] > 0
    breach = benchmark is not None and not benchmark["benchmark_pass"]
    config_present = REVIEWED_PATH.exists()
    if benchmark_blocked: status = "BLOCKED"
    elif breach: status = "NEEDS_ENGINE_CORRECTION"
    elif not commands_passed: status = "BLOCKED"
    elif summary["material_sampling_design_findings"]: status = "NEEDS_BOUNDS_REVIEW"
    elif benchmark["benchmark_pass"] and summary["audit_pass"] and config_present: status = "READY_FOR_SYNTHETIC_GENERATION"
    else: status = "BLOCKED"
    decision = {"status": status, "timestamp_utc": datetime.now(UTC).isoformat(), "decisive_evidence": {"benchmark_present": benchmark is not None, "benchmark_pass": None if benchmark is None else benchmark["benchmark_pass"],
                "reference_unreliable_count": None if benchmark is None else benchmark["reference"]["integration_failure_or_unreliable_count"],
                "benchmark_tolerance_breaches": breach, "audit_pass": summary["audit_pass"], "reviewed_sampling_config_present": config_present, "material_sampling_design_findings": summary["material_sampling_design_findings"],
                "external_validation_commands_passed": commands_passed},
                "remaining_limitations": ["Agreement is necessary but not proof of correctness.", "No teammate-equivalence claim.", "No real-market or NIFTY validation.", "No ANN training or large synthetic dataset occurred."]}
    write_json(FREEZE / "decision.json", decision)
    return decision


def run_audit(*, commands_passed: bool = False, freeze_only: bool = False) -> dict[str, Any]:
    if freeze_only:
        return _freeze(json.loads((OUTPUT / "bounds_audit_summary.json").read_text(encoding="utf-8")), commands_passed=commands_passed)
    bounds = _load_bounds()
    hard = bounds["hard_numerical_safety_bounds"]
    empirical = bounds["empirical_sampling_ranges"]["parameter_bounds"]
    sampler = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=SEED)
    unit = sampler.random(RAW_CANDIDATE_COUNT)
    lower = np.asarray([empirical[name]["lower"] for name in PARAMETER_NAMES])
    upper = np.asarray([empirical[name]["upper"] for name in PARAMETER_NAMES])
    raw = qmc.scale(unit, lower, upper)
    rows = [_margin_row(index, vector, hard) for index, vector in enumerate(raw)]
    frame = pd.DataFrame(rows).sort_values("candidate_id")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "sampled_parameter_vectors.csv", index=False)
    accepted = frame.loc[frame["accepted"]].copy()
    rejected = frame.loc[~frame["accepted"]].copy()
    accepted.to_csv(OUTPUT / "accepted_parameter_vectors.csv", index=False)
    rejected.to_csv(OUTPUT / "rejected_parameter_vectors.csv", index=False)
    reasons = rejected.assign(rejection_reason=rejected["rejection_reasons"].str.split(";")).explode("rejection_reason").groupby("rejection_reason", sort=True).size().rename("count").reset_index()
    reasons.to_csv(OUTPUT / "rejection_reasons.csv", index=False)
    accepted_count = len(accepted)
    rejected_count = len(rejected)
    proximity_definitions = [
        ("accepted_near_either_feller", "accepted_feller_near"),
        ("accepted_near_correlation_disk", "accepted_correlation_disk_near"),
        ("accepted_near_hard_bound", "accepted_hard_bound_near"),
        ("accepted_near_any_boundary", "accepted_any_boundary_near"),
        ("accepted_weak_slow_fast_separation", "accepted_weak_slow_fast_separation"),
    ]
    proximity = pd.DataFrame([{
        "metric": metric, "count": int(frame[column].sum()),
        "proportion": float(frame[column].sum() / accepted_count) if accepted_count else 0.0,
        "denominator": accepted_count, "population": "accepted_valid_candidates",
        "raw_candidate_count": len(frame), "accepted_candidate_count": accepted_count,
        "rejected_candidate_count": rejected_count, "rejection_rate": float(rejected_count / len(frame)),
    } for metric, column in proximity_definitions])
    proximity.to_csv(OUTPUT / "boundary_proximity.csv", index=False)
    selected = _select_priced(accepted)
    priced_rows: list[dict[str, Any]] = []
    surfaces: list[np.ndarray] = []
    for _, record in selected.iterrows():
        metrics = _surface_metrics(record[PARAMETER_NAMES].to_numpy(dtype=float))
        surfaces.append(metrics.pop("_surface"))
        priced_rows.append({**record.to_dict(), **metrics})
    priced = pd.DataFrame(priced_rows).sort_values("candidate_id") if priced_rows else pd.DataFrame(columns=["candidate_id"])
    nearest_pairs: list[dict[str, Any]] = []
    for left in range(len(surfaces)):
        for right in range(left + 1, len(surfaces)):
            rmse = float(np.sqrt(np.mean((surfaces[left] - surfaces[right]) ** 2)))
            a, b = selected.iloc[left], selected.iloc[right]
            normalized_distance = float(np.sqrt(np.mean([((a[name] - b[name]) / (upper[i] - lower[i])) ** 2 for i, name in enumerate(PARAMETER_NAMES)])))
            if rmse <= 1e-3 and normalized_distance >= .10:
                nearest_pairs.append({"candidate_id": int(a["candidate_id"]), "nearest_candidate_id": int(b["candidate_id"]), "surface_rmse": rmse, "normalized_parameter_distance": normalized_distance})
    if nearest_pairs:
        pairs = pd.DataFrame(nearest_pairs).sort_values(["candidate_id", "nearest_candidate_id", "surface_rmse"], kind="stable")
        nearest_by_id = pairs.groupby("candidate_id", as_index=False).first()
        priced = priced.merge(nearest_by_id, on="candidate_id", how="left")
    else:
        priced["nearest_candidate_id"], priced["surface_rmse"], priced["normalized_parameter_distance"] = np.nan, np.nan, np.nan
    pair_groups = {
        int(candidate_id): group[["nearest_candidate_id", "surface_rmse", "normalized_parameter_distance"]].to_dict(orient="records")
        for candidate_id, group in pairs.groupby("candidate_id", sort=True)
    } if nearest_pairs else {}
    priced["similar_surface_pair_count"] = priced["candidate_id"].map(lambda value: len(pair_groups.get(int(value), []))).astype(int)
    priced["similar_surface_pairs_json"] = priced["candidate_id"].map(
        lambda value: json.dumps(pair_groups.get(int(value), []), sort_keys=True, separators=(",", ":"))
    )
    priced.to_csv(OUTPUT / "priced_surface_summary.csv", index=False)
    quote_count = len(priced) * len(STRIKES) * len(MATURITIES) * 2
    finite_price_count = int(round(float(priced["finite_price_rate"].sum()) * len(STRIKES) * len(MATURITIES) * 2)) if len(priced) else 0
    finite_failures = quote_count - finite_price_count
    no_arb_failures = int((~priced["no_arbitrage_valid"]).sum()) if len(priced) else 0
    call_monotonicity_failures = int((~priced["call_strike_monotonicity"]).sum()) if len(priced) else 0
    put_monotonicity_failures = int((~priced["put_strike_monotonicity"]).sum()) if len(priced) else 0
    convexity_failures = int((~priced["strike_convexity"]).sum()) if len(priced) else 0
    atm_values = [value for serialized in priced["atm_normalized_call_by_maturity"] for value in json.loads(serialized).values()] if len(priced) else []
    surface_failures = finite_failures + no_arb_failures + call_monotonicity_failures + put_monotonicity_failures + convexity_failures
    proximity_metrics = {
        row["metric"]: {"count": int(row["count"]), "proportion": float(row["proportion"]),
                        "denominator": int(row["denominator"]), "population": row["population"]}
        for row in proximity.to_dict(orient="records")
    }
    rejection_rate = float(rejected_count / len(frame))
    accepted_boundary_rate = proximity_metrics["accepted_near_any_boundary"]["proportion"]
    material = bool(rejection_rate >= .25 or accepted_boundary_rate >= .10 or len(nearest_pairs) > 0 or surface_failures > 0)
    summary = {"schema_version": "1.0", "seed": SEED, "method": "scipy.stats.qmc.LatinHypercube", "raw_candidate_count": len(frame), "accepted_count": len(accepted), "rejected_count": len(rejected), "priced_surface_count": len(priced),
               "selection": "all accepted if <=250; otherwise ten equal-count rank strata by minimum normalized constraint distance with 25 seed-20260806 draws per stratum",
               "acceptance_rate": float(frame["accepted"].mean()), "rejection_rate": rejection_rate,
               "boundary_proximity": {"population": "accepted_valid_candidates", "denominator": accepted_count,
                                      "raw_candidate_count": len(frame), "accepted_candidate_count": accepted_count,
                                      "rejected_candidate_count": rejected_count, "rejection_rate": rejection_rate,
                                      "metrics": proximity_metrics},
               "priced_surface": {"quote_count": quote_count, "finite_price_count": finite_price_count,
                                  "finite_price_failure_count": finite_failures, "finite_price_rate": finite_price_count / quote_count if quote_count else None,
                                  "finite_price_rate_min_per_surface": float(priced["finite_price_rate"].min()) if len(priced) else None,
                                  "no_arbitrage_failures": no_arb_failures, "call_monotonicity_failures": call_monotonicity_failures,
                                  "put_monotonicity_failures": put_monotonicity_failures, "convexity_failures": convexity_failures,
                                  "minimum_normalized_price": float(priced["min_normalized_price"].min()) if len(priced) else None,
                                  "maximum_normalized_price": float(priced["max_normalized_price"].max()) if len(priced) else None,
                                  "minimum_atm_normalized_call": float(min(atm_values)) if atm_values else None,
                                  "maximum_atm_normalized_call": float(max(atm_values)) if atm_values else None,
                                  "term_structure_feature_variety_min": int(priced["term_structure_feature_variety"].min()) if len(priced) else None,
                                  "term_structure_feature_variety_max": int(priced["term_structure_feature_variety"].max()) if len(priced) else None,
                                  "runtime_total_seconds": float(priced["runtime_seconds"].sum()) if len(priced) else 0.0,
                                  "runtime_median_seconds": float(priced["runtime_seconds"].median()) if len(priced) else None,
                                  "runtime_p95_seconds": float(priced["runtime_seconds"].quantile(.95)) if len(priced) else None,
                                  "near_parameter_pairs_similar_surface_count": len(nearest_pairs), "implied_volatility": "unavailable/skipped: no validated inversion exists",
                                  "calendar_monotonicity_not_imposed": True},
               "identifiability_exposure_is_sampling_diagnostic_only": True, "does_not_claim_statistical_identifiability": True,
               "uniform_raw_bound_sampling_appropriate": not material, "material_sampling_design_findings": material,
               "audit_pass": bool(len(frame) == RAW_CANDIDATE_COUNT and len(priced) <= PRICE_LIMIT and surface_failures == 0)}
    write_json(OUTPUT / "bounds_audit_summary.json", summary)
    recommendations = [
        "# Parameter bounds audit recommendations", "",
        f"- REVIEW: provisional uniform sampling has acceptance rate {summary['acceptance_rate']:.4f}; it is {'not ' if not summary['uniform_raw_bound_sampling_appropriate'] else ''}appropriate without further evidence.",
        f"- REQUIRE_FINANCIAL_REVIEW: accepted-valid near-any-boundary rate is {accepted_boundary_rate:.4f} ({proximity_metrics['accepted_near_any_boundary']['count']}/{accepted_count}); no real-market calibration validates these ranges.",
        f"- SPLIT_SAMPLING_RANGE: accepted-valid near-Feller rate is {proximity_metrics['accepted_near_either_feller']['proportion']:.4f} ({proximity_metrics['accepted_near_either_feller']['count']}/{accepted_count}); retain boundary-near cases as challenge sampling rather than silently mixing them.",
        f"- REVIEW: {len(nearest_pairs)} priced parameter-pair comparisons met surface RMSE <= 1e-3 and normalized parameter distance >= 0.10; this is a sampling-design exposure, not a statistical-identifiability claim.",
        f"- KEEP: surface validity had {finite_failures} finite-price failures, {no_arb_failures} no-arbitrage failures, {call_monotonicity_failures} call-monotonicity failures, {put_monotonicity_failures} put-monotonicity failures, and {convexity_failures} convexity failures across {quote_count} prices.",
        "- KEEP: strict positivity, slow/fast ordering, strict Feller, individual correlation, joint-disk, and hard numerical-safety constraints remain enforced.",
        "- REVIEW: implied-volatility diagnostics are unavailable because this repository has no validated IV inversion.",
    ]
    (OUTPUT / "bounds_recommendations.md").write_text("\n".join(recommendations) + "\n", encoding="utf-8")
    _write_reviewed_config(bounds, summary)
    _freeze(summary, commands_passed=commands_passed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--commands-passed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_audit(commands_passed=args.commands_passed, freeze_only=args.freeze_only), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
