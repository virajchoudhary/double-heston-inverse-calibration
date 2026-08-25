"""Fast seal tests for the OOD evaluation infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import hashlib

from src.constants import PARAMETER_NAMES
from src.ood_boundary_evaluation import (
    aggregate_neural_seeds,
    degradation_against_id_baseline,
    COHORT_ROOT,
    RESEARCH_COHORT_AUTHORIZATION_PHRASE,
    ResearchEvaluationLocked,
    _validate_neural_payload,
    build_development_fixture,
    check_research_authorization,
    compare_replay,
    evaluate_prediction_matrix,
    freeze_identities,
    load_ready_config,
    bootstrap_materiality,
    load_reference_scaling,
    load_traditional_subset,
    materialize_traditional_subset,
    model3_readiness,
    checkpoint_readiness,
    prepare_freeze_identity,
    run_evaluation,
    verify_result_intake,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_cohort_structural_seal_is_exact() -> None:
    identity = freeze_identities()
    assert identity["counts"]["boundary_challenge"] == 120
    assert identity["counts"]["distribution_shift"] == 120
    assert identity["counts"]["maturity_conditioning_shift"] == 120
    assert identity["counts"]["incomplete_observation"] == 60
    assert identity["counts"]["serialized_research_total"] == 420
    assert identity["counts"]["clean_pricing_calls"] == 360
    assert identity["counts"]["pricing_failure_count"] == 0
    assert identity["replay_identical"] is True
    assert identity["research_model_metrics_present"] is False


def test_evaluation_config_pins_the_frozen_protocol() -> None:
    config = load_ready_config()
    assert config["status"] == "EVALUATION_INFRASTRUCTURE_SEALED_NO_RESULTS"
    assert config["research_execution_lock"][
        "hidden_environment_variable_sufficient"
    ] is False
    assert config["frozen_protocol"]["all_research_surfaces_sha256"] == (
        "e8b117ac93f6319e634fa28d6dd5ed884e86e130cf420e65eb8ef8da0276b7e4"
    )


def test_research_lock_requires_all_explicit_choices() -> None:
    with pytest.raises(ResearchEvaluationLocked):
        check_research_authorization(
            cohort="development_fixture", authorize=True, confirmation=""
        )
    with pytest.raises(ResearchEvaluationLocked):
        check_research_authorization(
            cohort="research", authorize=False, confirmation=""
        )
    with pytest.raises(ResearchEvaluationLocked):
        check_research_authorization(cohort="research", authorize=True, confirmation="yes")
    record = check_research_authorization(
        cohort="research",
        authorize=True,
        confirmation=RESEARCH_COHORT_AUTHORIZATION_PHRASE,
    )
    assert record["mechanism"] == "explicit_cohort_authorize_flag_exact_phrase"
    assert "AUTHORIZE" not in str(record)


def test_unauthorized_frozen_loader_is_locked_before_file_read(tmp_path) -> None:
    missing_manifest = tmp_path / "absent-identity.json"
    with pytest.raises(ResearchEvaluationLocked):
        from src.ood_boundary_evaluation import load_frozen_research_cohort

        load_frozen_research_cohort(authorized=False, identity_manifest_path=missing_manifest)
    assert not missing_manifest.exists()


def test_freeze_identity_manifest_pins_surface_order_without_outcomes(tmp_path) -> None:
    path = prepare_freeze_identity(tmp_path)
    payload = _json(path)
    assert payload["status"] == "FROZEN_OOD_STRUCTURAL_IDENTITY_VERIFIED"
    assert payload["result_status"] == "NO_METHOD_RESULTS_OPENED"
    assert len(payload["surface_id_order_sha256"]) == 64
    assert payload["counts"]["serialized_research_total"] == 420


def test_traditional_subset_is_deterministic_and_has_no_outputs(tmp_path) -> None:
    first = materialize_traditional_subset(tmp_path / "first")
    second = materialize_traditional_subset(tmp_path / "second")
    left = _json(first)
    right = _json(second)
    assert first.read_bytes() == second.read_bytes()
    assert left["total_selected"] == 60
    assert left["method_outputs_present"] is False
    counts = {cohort: 0 for cohort in (
        "boundary_challenge",
        "distribution_shift",
        "maturity_conditioning_shift",
        "incomplete_observation",
    )}
    for row in left["selections"]:
        counts[row["cohort"]] += 1
    assert counts == {
        "boundary_challenge": 15,
        "distribution_shift": 15,
        "maturity_conditioning_shift": 15,
        "incomplete_observation": 15,
    }
    assert left["selections"][0]["within_cohort_index"] == 0
    loaded = load_traditional_subset(first)
    assert len({row["surface_id"] for row in loaded}) == 60


def test_checkpoint_readiness_reports_all_missing_without_substitution(tmp_path) -> None:
    path = checkpoint_readiness(tmp_path)
    payload = _json(path)
    assert payload["status"] == "BLOCKED_CHECKPOINTS_NOT_AVAILABLE_LOCALLY"
    assert set(payload["missing_local_checkpoints"]) == {
        f"{method}/{seed}"
        for method in ("model1", "model2")
        for seed in (11, 22, 33)
    }
    assert payload["selection_policy"].startswith("all_seeds_or_method_is_blocked")


def test_model3_remains_waiting_without_fake_predictions(tmp_path) -> None:
    path = model3_readiness(tmp_path)
    payload = _json(path)
    assert payload["status"] == "WAITING_FOR_FROZEN_RESEARCH_CHECKPOINTS"
    assert payload["fake_predictions_allowed"] is False
    assert "No Stage-A execution" in payload["current_repository_truth"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 99),
        ("git_sha", "wrong"),
        ("run_kind", "SMOKE"),
        ("parameter_order", ["wrong"]),
    ],
)
def test_neural_payload_identity_fields_are_enforced(field: str, value: object) -> None:
    payload = {
        "spec": {"hidden_sizes": [1], "activation": "relu", "dropout": 0.0},
        "seed": 11,
        "run_kind": "RESEARCH",
        "git_sha": "expected",
        "parameter_order": list(PARAMETER_NAMES),
        "target_standardizer": {"mean": [0.0] * 10, "scale": [1.0] * 10},
    }
    payload[field] = value
    catalog = {
        "seed": 11,
        "training_git_sha": "expected",
        "expected_checkpoint_sha256": "hash",
        "actual_checkpoint_sha256": "hash",
        "identity_matches": True,
        "present": True,
    }
    with pytest.raises(Exception):
        _validate_neural_payload("model1", 11, payload, catalog)


def test_development_fixture_is_nonresearch_and_mask_safe(tmp_path) -> None:
    fixture_path, identity = build_development_fixture(tmp_path)
    payloads = [
        json.loads(line)
        for line in fixture_path.open(encoding="utf-8")
        if line.strip()
    ]
    assert identity["surface_count"] == 3
    assert identity["overlaps_frozen_research_surfaces"] is False
    incomplete = payloads[-1]
    parent = payloads[-2]
    assert incomplete["surface_id"] != parent["surface_id"]
    assert sum(incomplete["mask"]) == 12
    for price, mask, parent_price in zip(
        incomplete["prices"], incomplete["mask"], parent["prices"], strict=True
    ):
        if mask:
            assert price > 0.0 and price == parent_price
        else:
            assert price == 0.0
    assert all(
        payload["metadata"]["user_metadata"]["run_kind"]
        == "DEVELOPMENT_FIXTURE_NOT_RESEARCH_RESULT"
        for payload in payloads
    )


def test_development_fixture_ids_do_not_intersect_frozen_research(tmp_path) -> None:
    _, identity = build_development_fixture(tmp_path)
    fixture_path = tmp_path / "development_fixture_surfaces.jsonl"
    development_ids = {
        json.loads(line)["surface_id"]
        for line in fixture_path.open(encoding="utf-8")
        if line.strip()
    }
    research_ids = {
        json.loads(line)["surface_id"]
        for line in (COHORT_ROOT / "all_research_surfaces.jsonl").open(encoding="utf-8")
        if line.strip()
    }
    assert identity["surface_count"] == 3
    assert development_ids.isdisjoint(research_ids)


def test_metric_functions_are_reused_for_truth_pipeline(tmp_path) -> None:
    fixture_path, _ = build_development_fixture(tmp_path)
    from src.ood_boundary_evaluation import load_development_fixture

    cohort = load_development_fixture(fixture_path)
    scaling = load_reference_scaling()
    summary = evaluate_prediction_matrix(cohort, cohort.truths, scaling)
    assert summary["attempted_surface_count"] == 3
    assert summary["successful_prediction_count"] == 3
    assert summary["prediction_failure_count"] == 0
    assert summary["pricing_failure_count"] == 0
    assert summary["parameter_recovery"]["aggregate"][
        "range_scaled_parameter_rmse"
    ] == pytest.approx(0.0)
    assert summary["constraint_validity"]["constraint_validity_rate"] == 1.0


def test_degradation_rule_uses_frozen_thresholds() -> None:
    baseline = {
        "parameter_recovery": {
            "aggregate": {
                "range_scaled_parameter_rmse": 0.2,
                "standardized_parameter_rmse": 1.0,
            }
        },
        "clean_latent_repricing": {"normalized_price_rmse_mean": 0.001},
        "constraint_validity": {"constraint_validity_rate": 0.98},
    }
    ood = {
        "parameter_recovery": {
            "aggregate": {
                "range_scaled_parameter_rmse": 0.26,
                "standardized_parameter_rmse": 1.0,
            }
        },
        "clean_latent_repricing": {"normalized_price_rmse_mean": 0.001},
        "constraint_validity": {"constraint_validity_rate": 0.97},
    }
    result = degradation_against_id_baseline(ood, baseline)
    assert result["parameter_recovery.range_scaled_parameter_rmse"][
        "material"
    ] is True
    assert result["parameter_recovery.standardized_parameter_rmse"][
        "material"
    ] is False
    assert result["normalized_price_rmse_mean"]["degradation_ratio"] == pytest.approx(1.0)
    assert result["constraint_validity_failure_rate"]["absolute_increase"] == (
        pytest.approx(0.01)
    )
    assert result["decision"] == "MATERIAL_DEGRADATION_INDICATED"


def test_neural_seed_aggregation_retains_every_seed() -> None:
    predictions = {
        11: np.zeros((2, 10)),
        22: np.ones((2, 10)),
        33: np.full((2, 10), 2.0),
    }
    headline = {seed: {"value": float(seed)} for seed in predictions}
    aggregated = aggregate_neural_seeds(predictions, headline)
    assert aggregated["seeds"] == [11, 22, 33]
    assert aggregated["status"] == "ALL_SEEDS_RETAINED"
    assert aggregated["seed_mean_prediction"].shape == (2, 10)
    assert aggregated["stability"]["headline_metric_cross_seed_std"]["value"] == (
        pytest.approx(np.std([11.0, 22.0, 33.0], ddof=1))
    )


def test_bootstrap_uncertainty_uses_frozen_seed_and_interval() -> None:
    values = np.asarray([0.5, 2.0] * 20 + [0.5])
    result = bootstrap_materiality(values, 1.0)
    assert result["seed"] == 20260829
    assert result["resamples"] == 2000
    assert result["interval_spans_materiality"] is True
    assert result["ratio_interval"][0] <= result["point_ratio"] <= (
        result["ratio_interval"][1]
    )


def test_development_smoke_blocks_missing_checkpoints_and_rejects_partial_intake(
    tmp_path,
) -> None:
    result_directory = tmp_path / "smoke"
    result = run_evaluation(
        cohort_kind="development_fixture",
        methods=["truth_pipeline", "model1", "model2", "model3"],
        output_directory=result_directory,
        include_traditional=True,
        traditional_workers=1,
        max_nfev_override=1,
    )
    assert result["status"] == "PARTIAL_OR_BLOCKED"
    assert result["method_statuses"]["model1"]["status"] == (
        "NOT_AVAILABLE_BLOCKED_CHECKPOINTS"
    )
    assert result["method_statuses"]["model2"]["status"] == (
        "NOT_AVAILABLE_BLOCKED_CHECKPOINTS"
    )
    assert result["method_statuses"]["model3"]["status"] == (
        "WAITING_FOR_FROZEN_RESEARCH_CHECKPOINTS"
    )
    manifest_path = result_directory / "result_manifest.json"
    manifest = _json(manifest_path)
    assert manifest["run_kind"] == "development_fixture"
    assert manifest["authorization_record"] is None
    research_ids = {
        line.strip()
        for line in (COHORT_ROOT / "all_research_surfaces.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    }
    prediction_rows = []
    for path in result_directory.glob("*_predictions.csv"):
        prediction_rows.extend(pd.read_csv(path).to_dict(orient="records"))
    assert prediction_rows
    assert all(row["surface_id"] not in research_ids for row in prediction_rows)
    intake = verify_result_intake(result_directory, require_complete=False)
    assert intake["hashes_verified"] is True
    with pytest.raises(Exception, match="partial/blocked/inconclusive result refused"):
        verify_result_intake(result_directory, require_complete=True)


def test_development_smoke_core_artifacts_replay_identically(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_evaluation(
        cohort_kind="development_fixture",
        methods=["truth_pipeline"],
        output_directory=first,
    )
    run_evaluation(
        cohort_kind="development_fixture",
        methods=["truth_pipeline"],
        output_directory=second,
    )
    replay = compare_replay(first, second)
    assert replay["deterministic_core_identical"] is True
    assert "_runtime.json" not in "".join(replay["comparisons"])


def test_result_intake_rejects_prediction_row_misalignment(tmp_path) -> None:
    source = Path("evidence/ood_boundary_development_smoke_v1")
    destination = tmp_path / "tampered"
    import shutil

    shutil.copytree(source, destination)
    prediction_path = next(destination.glob("truth_pipeline_predictions.csv"))
    frame = pd.read_csv(prediction_path)
    frame.loc[0, "surface_id"] = "MISALIGNED"
    manifest = json.loads((destination / "result_manifest.json").read_text())
    manifest["evaluation_config_sha256"] = hashlib.sha256(
        Path("configs/ood_boundary_evaluation_ready.yaml").read_bytes()
    ).hexdigest()
    manifest["implementation_hashes"] = {
        "harness_source_sha256": "recorded",
        "cli_source_sha256": "recorded",
        "evaluation_config_sha256": "recorded",
    }
    frame.to_csv(prediction_path, index=False, lineterminator="\n")
    relative = prediction_path.relative_to(destination).as_posix()
    manifest["artifact_hashes"][relative] = hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    (destination / "result_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="prediction row misalignment"):
        verify_result_intake(destination, require_complete=False)


def test_truth_pipeline_can_never_open_research(tmp_path) -> None:
    with pytest.raises(Exception, match="truth_pipeline is forbidden"):
        run_evaluation(
            cohort_kind="research",
            methods=["truth_pipeline"],
            output_directory=tmp_path,
            authorize=True,
            confirmation=RESEARCH_COHORT_AUTHORIZATION_PHRASE,
        )


def test_empty_method_list_cannot_be_complete(tmp_path) -> None:
    with pytest.raises(Exception, match="at least one method"):
        run_evaluation(
            cohort_kind="development_fixture",
            methods=[],
            output_directory=tmp_path / "empty",
        )


def test_output_refuses_frozen_cohort_tree() -> None:
    with pytest.raises(Exception, match="refusing to write evaluation output"):
        run_evaluation(
            cohort_kind="development_fixture",
            methods=["truth_pipeline"],
            output_directory=COHORT_ROOT / "must_not_write",
        )
