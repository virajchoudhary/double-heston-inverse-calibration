from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ann_dataset_FIRST_RESEARCH.yaml"
EVIDENCE = ROOT / "outputs" / "core_dataset_readiness" / "core_dataset_readiness.json"


def test_deprecated_historical_core_dataset_config_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    superseded = config["superseded_by"]
    assert config["status"] == "HISTORICAL_DEPRECATED_SUPERSEDED_BY_R2_FINAL_CONTRACT"
    assert superseded["active_config"] == "configs/r2_synthetic_generation_FINAL.yaml"
    assert superseded["representation"] == "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
    assert superseded["historical_note"] == "retained_verbatim_as_evidence_that_108_was_once_planned"
    representation = config["ann_representation"]
    assert representation["log_moneyness"] == [-0.30, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30]
    assert representation["maturity_days"] == [7, 14, 30, 60, 90, 180]
    assert representation["input_dimension"] == 108


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
