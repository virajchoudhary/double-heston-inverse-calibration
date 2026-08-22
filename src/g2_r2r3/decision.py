"""Frozen decision evaluator for the G2 R2-vs-R3 study.

Applies the predeclared decision rule from
``docs/G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md`` EXACTLY once, at the end,
using only thresholds imported from ``frozen.py``.  The evaluator accepts no
threshold arguments at runtime; there is no mechanism to change protocol
thresholds post hoc (test-enforced).
"""

from __future__ import annotations

from typing import Any

from . import frozen


def _improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate <= 0.0 else float("-inf")
    return 1.0 - candidate / baseline


def comparative_assessment(
    r2_metrics: dict[str, float], r3_metrics: dict[str, float]
) -> dict[str, Any]:
    """Existing interpretation bands applied to dispersion and cluster evidence."""
    median_improvement = _improvement(
        r2_metrics["median_dispersion"], r3_metrics["median_dispersion"]
    )
    maximum_improvement = _improvement(
        r2_metrics["maximum_dispersion"], r3_metrics["maximum_dispersion"]
    )
    cluster_change = r3_metrics["mean_cluster_count"] - r2_metrics["mean_cluster_count"]
    strong = (
        median_improvement >= frozen.STRONG_IMPROVEMENT_MEDIAN
        and maximum_improvement >= frozen.STRONG_IMPROVEMENT_MAXIMUM
        and cluster_change < 0
    )
    partial = (
        median_improvement >= frozen.PARTIAL_IMPROVEMENT_MEDIAN
        and maximum_improvement >= frozen.PARTIAL_IMPROVEMENT_MAXIMUM
        and cluster_change <= 0
    )
    if strong:
        classification = "STRONG_IMPROVEMENT"
    elif partial:
        classification = "PARTIAL_IMPROVEMENT"
    else:
        classification = "NO_MATERIAL_IMPROVEMENT"
    return {
        "median_dispersion_improvement": median_improvement,
        "maximum_dispersion_improvement": maximum_improvement,
        "r2_median_dispersion": r2_metrics["median_dispersion"],
        "r3_median_dispersion": r3_metrics["median_dispersion"],
        "r2_maximum_dispersion": r2_metrics["maximum_dispersion"],
        "r3_maximum_dispersion": r3_metrics["maximum_dispersion"],
        "r2_mean_cluster_count": r2_metrics["mean_cluster_count"],
        "r3_mean_cluster_count": r3_metrics["mean_cluster_count"],
        "cluster_count_change": cluster_change,
        "classification": classification,
        "thresholds_locked_to_frozen_module": True,
    }


def practical_non_identifiability(
    selected_metrics_by_noise: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Predeclared operationalization (see ``frozen.py`` docstring)."""
    level = frozen.NON_IDENTIFIABILITY_NOISE_LEVEL
    metrics = selected_metrics_by_noise[f"{level:.4f}"]
    parameter_test = (
        metrics["median_best_parameter_rmse_scaled"]
        > frozen.MATERIAL_DISPLACEMENT_RMSE
    )
    repricing_test = (
        metrics["median_best_repricing_rmse_relative"]
        <= frozen.NON_IDENTIFIABILITY_REPRICING_FACTOR * level
    )
    retained = bool(parameter_test and repricing_test)
    return {
        "noise_level": level,
        "median_best_parameter_rmse_scaled": metrics[
            "median_best_parameter_rmse_scaled"
        ],
        "material_displacement_threshold": frozen.MATERIAL_DISPLACEMENT_RMSE,
        "median_best_repricing_rmse_relative": metrics[
            "median_best_repricing_rmse_relative"
        ],
        "repricing_ceiling": frozen.NON_IDENTIFIABILITY_REPRICING_FACTOR * level,
        "parameter_displacement_test_passed": bool(parameter_test),
        "repricing_at_noise_scale_test_passed": bool(repricing_test),
        "practical_non_identifiability_retained": retained,
    }


def apply_frozen_decision(
    hard_requirements: dict[str, dict[str, Any]],
    assessment: dict[str, Any],
    non_identifiability_by_representation: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the predeclared freeze rule once, in order, unchanged.

    Rule 1: R3 satisfies hard market-construction requirements AND shows
    strong or partial improvement without violating an applicable holdout
    guardrail -> freeze R3.
    Rule 2: otherwise freeze R2 provided R2 satisfies the hard requirements.
    Rule 3: if both remain practically non-identifiable at realistic noise,
    do not reopen representation search; retain the finding.
    Rule 4: only fail G2 if BOTH candidates fail the hard requirements.

    ``non_identifiability_by_representation`` supplies the predeclared test
    outcome per candidate; the evaluator itself reads the SELECTED
    representation's record, so no caller pre-guesses the winner.
    """
    r2_eligible = bool(hard_requirements["R2"]["satisfied"])
    r3_eligible = bool(hard_requirements["R3"]["satisfied"])
    holdout_applicable = False  # frozen: NOT_APPLICABLE per protocol section 7
    holdout_violated = False

    improvement = assessment["classification"] in (
        "STRONG_IMPROVEMENT",
        "PARTIAL_IMPROVEMENT",
    )
    if not r2_eligible and not r3_eligible:
        # Rule 4: both candidates failed the hard market-construction
        # requirements.  The protocol requires exactly the FAILED label here;
        # no PASSED label may be derived from either candidate's records.
        return {
            "selected_representation": None,
            "g2_label": frozen.G2_LABEL_FAILED_MARKET_CONSTRUCTION,
            "rule_applied": "rule_4_both_failed_hard_requirements",
            "improvement_classification": assessment["classification"],
            "practical_non_identifiability_retained": None,
            "practical_non_identifiability_by_representation": (
                non_identifiability_by_representation
            ),
            "holdout_guardrail": frozen.HOLDOUT_GUARDRAIL_STATUS,
        }
    if r3_eligible and improvement and not (holdout_applicable and holdout_violated):
        selected = "R3"
        rule = "rule_1_r3_market_supported_with_improvement"
    elif r2_eligible:
        selected = "R2"
        rule = "rule_2_r2_simpler_market_supported_representation"
    else:
        selected = "R3"
        rule = "rule_2_fallback_r3_only_eligible_candidate"

    non_ident = non_identifiability_by_representation[selected]

    label = (
        frozen.G2_LABEL_PRACTICAL_NON_IDENTIFIABILITY
        if non_ident["practical_non_identifiability_retained"]
        else frozen.G2_LABEL_IDENTIFIABILITY_ACCEPTABLE
    )
    # Rule 4 is unreachable here: it returned the exact FAILED label above.
    return {
        "selected_representation": selected,
        "g2_label": label,
        "rule_applied": rule,
        "improvement_classification": assessment["classification"],
        "practical_non_identifiability_retained": non_ident[
            "practical_non_identifiability_retained"
        ],
        "practical_non_identifiability_selected_representation": non_ident,
        "holdout_guardrail": frozen.HOLDOUT_GUARDRAIL_STATUS,
    }
