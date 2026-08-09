from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ann_dataset_FIRST_RESEARCH.yaml"
EVIDENCE = ROOT / "outputs" / "core_dataset_readiness" / "core_dataset_readiness.json"


def test_prepared_core_dataset_config_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    readiness = config["readiness"]
    assert config["status"] == "PREPARED_NOT_EXECUTED"
    assert readiness["core_dataset_ready"] is True
    assert readiness["core_dataset_ready_meaning"] == "ready_to_generate_not_already_generated"
    assert readiness["core_generation_plan_ready"] is True
    assert readiness["core_dataset_generated"] is False
    assert readiness["challenge_stress_ready"] is False
    assert readiness["engine_review_required"] is False
    assert readiness["global_reviewed_sampling_audit_status"] == "NEEDS_SAMPLER_CORRECTION"
    assert readiness["global_reviewed_sampling_audit_pass"] is False
    assert readiness["scientific_generation_authorized"] is True
    assert readiness["dataset_generation_execution_authorized"] is False
    assert readiness["training_authorized"] is False
    assert config["implementation_boundary"]["reviewed_core_generator_implemented"] is False
    assert config["implementation_boundary"]["existing_generic_generator_compatible"] is False
    assert not (ROOT / "src" / "generate_reviewed_core_dataset.py").exists()
    assert config["synthetic_market_state"] == {"spot": 100.0, "risk_free_rate": 0.02, "dividend_yield": 0.01}
    representation = config["ann_representation"]
    assert representation["target"] == "normalized_price"
    assert representation["option_types"] == ["call", "put"]
    assert representation["log_moneyness"] == [-0.30, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30]
    assert representation["maturity_days"] == [7, 14, 30, 60, 90, 180]
    assert representation["input_dimension"] == 108
    assert representation["target_dimension"] == 10
    assert representation["parameter_order"] == [
        "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
        "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast",
    ]
    plan = config["clean_surface_plan"]["normal_clean_surface_quotas"]
    assert plan["train"] == {
        "total": 7500,
        "interior_train": 6250,
        "wide_valid_train": 1250,
    }
    assert plan["validation"] == {
        "total": 1250,
        "interior_train": 1042,
        "wide_valid_train": 208,
    }
    assert plan["test"] == {
        "total": 1250,
        "interior_train": 1042,
        "wide_valid_train": 208,
    }
    assert sum(part["total"] for part in (plan["train"], plan["validation"], plan["test"])) == 10000
    assert sum(plan[split]["interior_train"] for split in ("train", "validation", "test")) == 8334
    assert sum(plan[split]["wide_valid_train"] for split in ("train", "validation", "test")) == 1666
    assert plan["distribution_totals"] == {"interior_train": 8334, "wide_valid_train": 1666, "total": 10000}
    assert config["clean_surface_plan"]["composition_percent"] == {"interior_train": 83.34, "wide_valid_train": 16.66}
    assert config["clean_surface_plan"]["approximate_interior_to_wide_ratio"] == "5:1"
    integrity = config["clean_surface_plan"]["split_integrity"]
    assert integrity == {
        "unit": "whole_surface",
        "no_quote_row_splitting": True,
        "unseen_parameter_vectors_across_splits": True,
        "no_surface_id_overlap": True,
        "no_parameter_vector_overlap": True,
        "fit_normalization_on_train_only": True,
    }
    assert config["clean_surface_plan"]["excluded_populations"] == ["boundary_challenge", "ood_test", "noisy_surfaces"]
    assert config["noise_follow_on_plan"]["levels"] == [0, 0.005, 0.01, 0.02]
    assert config["noise_follow_on_plan"]["retain_metadata"] == ["original_clean_surface_id", "noise_level", "noise_seed", "distribution_identity"]
    assert config["noise_follow_on_plan"]["no_clipping_projection_or_discard"] is True
    assert config["noise_follow_on_plan"]["no_arbitrage_repair"] is True
    assert config["noise_follow_on_plan"]["participates_in_core_ready_gate"] is False
    assert config["runtime_estimate"] == {
        "basis": "verified_price_only_108_quote_surface_timing",
        "surfaces": 10000,
        "mean_minutes": 16.04,
        "p95_minutes": 16.34,
        "excludes": ["selection", "validation", "serialization", "hashing", "retries", "contention"],
    }
    assert config["next_milestone"] == {
        "name": "implement_and_test_reviewed_core_generator",
        "generator_exists": False,
        "future_command": None,
        "dataset_generation_authorized_in_this_task": False,
    }


def test_machine_readable_readiness_evidence_contract() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert evidence["baseline_commit"] == "bdb5aa275ec6a2b91841a07fd4f9f7fea42a1dca"
    assert evidence["production_source_checksum"]["sha256"] == "154d53fba17caddcb4d5a8d72001a43aa3cbb90cb1d6d9177c07ddefc5414b0a"
    pricing_source = ROOT / evidence["production_source_checksum"]["path"]
    assert hashlib.sha256(pricing_source.read_bytes()).hexdigest() == evidence["production_source_checksum"]["sha256"]
    assert evidence["readiness"]["core_dataset_ready"] is True
    assert evidence["readiness"]["challenge_stress_ready"] is False
    assert evidence["readiness"]["global_reviewed_sampling_audit_status"] == "NEEDS_SAMPLER_CORRECTION"
    assert evidence["readiness"]["global_reviewed_sampling_audit_pass"] is False
    assert evidence["decision"]["CORE_DATASET_READY"] is True
    assert evidence["decision"]["CHALLENGE_STRESS_READY"] is False
    assert evidence["implementation_boundary"]["reviewed_core_generator_implemented"] is False
    assert evidence["implementation_boundary"]["existing_generic_generator_compatible"] is False
    assert not (ROOT / "src" / "generate_reviewed_core_dataset.py").exists()
    for path, expected_hash in evidence["source_hashes_verified"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected_hash
    plan = evidence["first_dataset_plan"]
    assert plan["train"] == {"total": 7500, "interior_train": 6250, "wide_valid_train": 1250}
    assert plan["validation"] == {"total": 1250, "interior_train": 1042, "wide_valid_train": 208}
    assert plan["test"] == {"total": 1250, "interior_train": 1042, "wide_valid_train": 208}
    assert sum(part["total"] for part in (plan["train"], plan["validation"], plan["test"])) == 10000
    assert sum(plan[split]["interior_train"] for split in ("train", "validation", "test")) == 8334
    assert sum(plan[split]["wide_valid_train"] for split in ("train", "validation", "test")) == 1666
    assert plan["distribution_totals"] == {"interior_train": 8334, "wide_valid_train": 1666}
    assert plan["composition_percent"] == {"interior_train": 83.34, "wide_valid_train": 16.66}
    assert plan["approximate_interior_to_wide_ratio"] == "5:1"
    assert plan["input_dimension"] == 108
    assert plan["target_dimension"] == 10
    assert evidence["ann_representation"] == config["ann_representation"]
    assert plan["input_dimension"] == config["ann_representation"]["input_dimension"]
    assert plan["target_dimension"] == config["ann_representation"]["target_dimension"]
    yaml_plan = config["clean_surface_plan"]
    assert plan["composition_percent"] == yaml_plan["composition_percent"]
    assert plan["approximate_interior_to_wide_ratio"] == yaml_plan["approximate_interior_to_wide_ratio"]
    assert plan["train"] == yaml_plan["normal_clean_surface_quotas"]["train"]
    assert plan["validation"] == yaml_plan["normal_clean_surface_quotas"]["validation"]
    assert plan["test"] == yaml_plan["normal_clean_surface_quotas"]["test"]
    assert plan["distribution_totals"] == {
        key: value
        for key, value in yaml_plan["normal_clean_surface_quotas"]["distribution_totals"].items()
        if key != "total"
    }
    assert plan["split_integrity"] == {
        "atomic_unit": "complete_parameter_vector_and_surface",
        "no_quote_row_splitting": True,
        "no_surface_id_overlap": True,
        "no_parameter_vector_overlap": True,
        "fit_normalization_on_train_only": True,
    }
    assert plan["excluded_populations"] == ["boundary_challenge", "ood_test", "noisy_surfaces"]
    assert plan["generated"] is False and plan["training_started"] is False
    assert evidence["ood_evidence"]["train_validation_assignments"] == 0
    assert evidence["ood_evidence"]["kappa_fast_ood_observed_min"] > evidence["ood_evidence"]["kappa_fast_normal_observed_max"]
    assert [case["key"] for case in evidence["retained_challenge_cases"]] == [["boundary_challenge", item] for item in (1043, 1091, 1180, 1276)]
    assert all(case["all_96_node_gates_pass"] and case["adaptive_reference_reliable"] for case in evidence["retained_challenge_cases"])
    assert all(case["adaptive_reference_warning_count"] == 0 and case["engine_review_required"] is False for case in evidence["retained_challenge_cases"])
    assert all(
        case["split"] == "challenge_excluded"
        and case["classification"] == "numerical_tolerance_stress"
        for case in evidence["retained_challenge_cases"]
    )
    assert evidence["noise_evidence"]["levels"] == [0.0, 0.005, 0.01, 0.02]
    assert evidence["noise_evidence"]["seeds"] == [20260807, 20265810, 20270813, 20275816]
    assert evidence["noise_evidence"]["retain_metadata"] == ["original_clean_surface_id", "noise_level", "noise_seed", "distribution_identity"]
    assert evidence["noise_evidence"]["no_clipping_projection_or_discard"] is True
    assert evidence["noise_evidence"]["no_arbitrage_repair"] is True
    assert evidence["noise_evidence"]["raw_shape_failures_invalidate_core_readiness"] is False
    assert evidence["noise_evidence"]["ready_gating"] is False
    assert evidence["next_milestone"]["generator_exists"] is False
    assert evidence["next_milestone"]["future_command"] is None
