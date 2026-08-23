"""Frozen scientific-contract checks for final R2 synthetic generation."""

from __future__ import annotations

import math
from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd

import pytest
import yaml

from src.constants import PARAMETER_NAMES
from src.r2_representation import surface_from_vectors
from src import r2_synthetic_generation as generator


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "r2_synthetic_generation_FINAL.yaml"
LEGACY_CONFIG_PATH = ROOT / "configs" / "ann_dataset_FIRST_RESEARCH.yaml"
CONTRACT_PATH = ROOT / "docs" / "R2_SYNTHETIC_GENERATION_CONTRACT.md"
FREEZE_MARKER_PATH = ROOT / "evidence" / "R2_CONTRACT_FROZEN_BEFORE_PILOT.txt"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_generation_identity_and_r2_representation(config: dict) -> None:
    assert config["schema_version"] == "1.0"
    assert config["contract_name"] == "R2_SYNTHETIC_GENERATION_FINAL"
    assert config["generator_version"] == "r2-synthetic-generation-v1"
    assert config["status"] == "FROZEN_BEFORE_PILOT"
    assert config["tracking_issue"] == 27
    assert config["final_10k_generated"] is False

    representation = config["representation"]
    assert representation["name"] == "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
    assert representation["version"] == "1.0"
    assert representation["nominal_slot_count"] == 20
    assert representation["expiry_ranks"] == [1, 2]
    assert representation["target_log_moneyness"] == [-0.10, -0.05, 0.00, +0.05, +0.10]
    assert representation["option_types"] == ["call", "put"]
    assert (
        representation["slot_order"]
        == "option_type_major_then_expiry_rank_then_log_moneyness"
    )
    assert representation["implementation_package"] == "src/r2_representation"
    assert representation["synthetic_mask_policy"] == "all_true_complete_by_construction"
    assert representation["price_normalization"] == "spot_normalized_with_spot_100"


def test_canonical_parameter_and_reviewed_sampling_contract(config: dict) -> None:
    expected_parameters = [
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
    parameter_contract = config["parameter_contract"]
    assert list(parameter_contract["order"]) == expected_parameters
    assert (
        parameter_contract["reviewed_sampling_config"]
        == "configs/parameter_sampling_REVIEWED.yaml"
    )
    assert (
        parameter_contract["transforms"]
        == "reuse_reviewed_latent_lhs_conditional_transforms_unchanged"
    )
    assert (
        parameter_contract["hard_constraints"]
        == "reuse_reviewed_hard_constraints_unchanged"
    )
    assert (
        parameter_contract["acceptance_margin_policy"]
        == "reuse_reviewed_distribution_acceptance_gate_unchanged"
    )
    assert parameter_contract["canonical_validation"] == "src.constraints.validate_parameters"

    sampling = config["sampling"]
    assert sampling["method"] == "scipy_stats_qmc_latin_hypercube_latent_coordinates"
    assert sampling["candidate_id_rule"] == "zero_based_row_order_within_distribution"
    assert sampling["selection_order"] == "distribution_then_candidate_id_ascending"
    assert sampling["fixed_pool_policy"] == (
        "generate_entire_pool_retain_all_candidates_and_rejections_no_refill_no_reseed"
    )
    assert sampling["insufficient_pool_behavior"] == "HARD_FAIL"


def test_exact_final_quotas_pools_and_seeds(config: dict) -> None:
    final = config["quotas"]["final_clean_core"]
    assert final["total_surfaces"] == 10_000
    assert final["distributions"] == {
        "interior_train": 8_334,
        "wide_valid_train": 1_666,
    }
    assert final["splits"] == {
        "train": {"total": 7_500, "interior_train": 6_250, "wide_valid_train": 1_250},
        "validation": {"total": 1_250, "interior_train": 1_042, "wide_valid_train": 208},
        "test": {"total": 1_250, "interior_train": 1_042, "wide_valid_train": 208},
    }

    pilot = config["quotas"]["development_pilot_not_final_research_dataset"]
    assert pilot["total_surfaces"] == 240
    assert pilot["distributions"] == {"interior_train": 200, "wide_valid_train": 40}
    assert pilot["splits"] == {
        "train": {"total": 180, "interior_train": 150, "wide_valid_train": 30},
        "validation": {"total": 30, "interior_train": 25, "wide_valid_train": 5},
        "test": {"total": 30, "interior_train": 25, "wide_valid_train": 5},
    }

    pools = config["sampling"]["pools"]
    seeds = config["sampling"]["seeds"]
    assert pools["pilot"] == {
        "interior_train": {"candidate_count": 400, "required_quota": 200},
        "wide_valid_train": {"candidate_count": 200, "required_quota": 40},
    }
    assert pools["final"] == {
        "interior_train": {"candidate_count": 15_000, "required_quota": 8_334},
        "wide_valid_train": {"candidate_count": 5_000, "required_quota": 1_666},
    }
    assert seeds == {
        "pilot_interior_train": 20260822,
        "pilot_wide_valid_train": 20260823,
        "final_interior_train": 20260807,
        "final_wide_valid_train": 20260808,
    }
    for cohort in ("pilot", "final"):
        for distribution in ("interior_train", "wide_valid_train"):
            spec = pools[cohort][distribution]
            quota_key = (
                "development_pilot_not_final_research_dataset"
                if cohort == "pilot"
                else "final_clean_core"
            )
            assert spec["required_quota"] == config["quotas"][quota_key]["distributions"][distribution]


def test_split_assignment_is_deterministic_and_atomic(config: dict) -> None:
    split = config["split_assignment"]
    assert split["algorithm"] == (
        "accepted_candidates_sorted_by_candidate_id_exact_nonoverlapping_slices_per_distribution"
    )
    assert split["random_permutation_used"] is False
    assert split["atomic_unit"] == "whole_parameter_vector_and_whole_surface"
    assert split["quote_row_splitting_allowed"] is False
    assert split["cross_split_parameter_overlap_allowed"] is False
    assert split["duplicate_parameter_vectors_allowed"] is False


def test_synthetic_only_conditioning_support_and_mapping(config: dict) -> None:
    conditioning = config["conditioning"]
    assert conditioning["classification"] == (
        "SYNTHETIC_ONLY_ENGINEERING_RESEARCH_DESIGN_NOT_REAL_MARKET_DISTRIBUTION_CLAIM"
    )
    assert conditioning["real_market_inputs_used"] is False
    assert conditioning["spot"] == 100.0
    assert conditioning["deterministic_mode"] == "predeclared_lattice_mixed_radix_stride"
    assert conditioning["seeds"] == {"pilot": 20260822, "final": 20260807}
    assert conditioning["strides"] == {"pilot": 997, "final": 1103}

    lattice = conditioning["lattice"]
    rank1 = lattice["rank1_dte_days"]
    gaps = lattice["rank2_gap_dte_days"]
    rates = lattice["rates"]
    offsets = lattice["carry_offsets"]
    assert rank1 == [7, 14, 21, 30, 45, 60, 75, 90]
    assert gaps == [7, 14, 21, 30, 45, 60, 90]
    assert rates == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    assert offsets == [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03]
    assert lattice["combination_count"] == len(rank1) * len(gaps) * len(rates) * len(offsets)
    assert all(gap > 0 for gap in gaps)
    assert max(rank1) + max(gaps) == 180
    assert all(math.gcd(stride, lattice["combination_count"]) == 1 for stride in conditioning["strides"].values())
    assert "modulo" in lattice or "modulo" in conditioning["index_formula"]["lattice_index"]
    assert conditioning["invariants"] == {
        "dte_integer_days": True,
        "rank1_dte_positive": True,
        "rank2_dte_strictly_greater_than_rank1": True,
        "rates_finite": True,
        "carries_finite": True,
        "observed_nse_option_spot_future_rate_or_carry_used": False,
    }


def test_production_pricing_noise_and_failure_gates_are_closed(config: dict) -> None:
    pricing = config["pricing"]
    assert pricing["production_source"] == "src/double_heston.py"
    assert pricing["entrypoint"] == "price_double_heston_surface"
    assert pricing["node_count"] == 64
    assert pricing["target_strike_rule"] == "spot_exp_target_log_moneyness"
    assert pricing["rank_pricing"] == "constant_conditioning_per_rank_piece"
    assert pricing["clipping_or_imputation_allowed"] is False
    assert pricing["replacement_on_failure_allowed"] is False

    noise = config["noise"]
    assert noise["clean_core_level"] == 0.0
    assert noise["future_levels_are_separate_derivatives"] == [0.005, 0.01, 0.02]
    assert noise["generation_in_this_milestone"] is False

    exclusions = config["exclusions"]
    assert exclusions["clean_core_excludes"] == [
        "boundary_challenge",
        "ood_test",
        "noisy_copies",
        "real_market_data",
    ]
    assert exclusions["global_sampler_status_preserved"] == "NEEDS_SAMPLER_CORRECTION"
    assert exclusions["challenge_stress_ready"] is False

    failures = config["failure_retention"]
    assert failures["pricing_failure_policy"] == (
        "retain_candidate_surface_id_parameters_conditioning_and_error_then_fail_closed"
    )
    assert failures["rejection_policy"] == "retain_every_candidate_and_all_rejection_reasons"
    assert failures["silent_replacement_or_refill_forbidden"] is True


def test_execution_gates_prohibit_final_generation_and_training(config: dict) -> None:
    gates = config["execution_gates"]
    assert gates["contract_freeze_before_pilot_output"] == "REQUIRED"
    assert gates["pilot_command_requires_contract_checkpoint"] is True
    assert gates["readiness_requires_verified_pilot"] is True
    assert gates["final_readiness_command"] == (
        "python -m src.r2_synthetic_generation readiness "
        "--output evidence/final_r2_candidate_pool_readiness_20260822"
    )
    assert gates["final_10k_generation_command"] == "NOT_AUTHORIZED_IN_THIS_MILESTONE"
    assert gates["final_10k_requires_separate_explicit_command_and_authorization"] is True
    assert gates["training_commands_in_this_milestone"] == "NONE"
    assert config["final_10k_generated"] is False


def test_freeze_marker_precedes_any_authoritative_pilot() -> None:
    assert FREEZE_MARKER_PATH.is_file()
    text = FREEZE_MARKER_PATH.read_text(encoding="utf-8")
    assert "CONTRACT_FROZEN_BEFORE_PILOT" in text
    assert "NO AUTHORITATIVE PILOT OUTPUT EXISTED WHEN THIS CONTRACT SHA WAS COMMITTED" in text


def test_legacy_108_plan_is_historical_not_active() -> None:
    legacy = yaml.safe_load(LEGACY_CONFIG_PATH.read_text(encoding="utf-8"))
    assert legacy["status"] == "HISTORICAL_DEPRECATED_SUPERSEDED_BY_R2_FINAL_CONTRACT"
    superseded = legacy["superseded_by"]
    assert superseded["active_config"] == "configs/r2_synthetic_generation_FINAL.yaml"
    assert superseded["representation"] == "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
    assert superseded["reason"] == (
        "legacy_fixed_calendar_108_grid_is_not_the_active_final_representation"
    )
    representation = legacy["ann_representation"]
    assert representation["input_dimension"] == 108
    assert representation["log_moneyness"] == [
        -0.30, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30
    ]
    assert representation["maturity_days"] == [7, 14, 30, 60, 90, 180]


def test_contract_document_states_material_boundaries() -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    for required in (
        "FROZEN_BEFORE_PILOT",
        "FINAL_10K_NOT_GENERATED",
        "ANN_RESEARCH_TRAINING = NOT_STARTED",
        "No range or\nthreshold is tuned by this milestone.",
        "There is no refill loop, second seed, range change,\nthreshold change, or replacement of difficult selected rows.",
        "No observed NSE option price",
        "DTE2\nis always strictly greater than DTE1",
        "Prices come only from unchanged production source",
        "A pricing or numerical failure preserves the candidate identity",
        "The contract checkpoint must be committed before any pilot output exists.",
    ):
        assert required in text


def test_generator_loads_and_validates_frozen_contract(config: dict) -> None:
    assert generator.load_generation_config() == config


def test_cohort_quota_lookup_maps_pilot_to_development_contract(config: dict) -> None:
    assert generator.config_quotas("pilot") == config["quotas"]["development_pilot_not_final_research_dataset"]
    assert generator.config_quotas("final") == config["quotas"]["final_clean_core"]


def test_generator_rejects_quota_drift(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    drifted = copy.deepcopy(config)
    drifted["quotas"]["development_pilot_not_final_research_dataset"]["total_surfaces"] = 241
    monkeypatch.setattr(generator.yaml, "safe_load", lambda _path: drifted)
    with pytest.raises(generator.GenerationContractError, match="quota contract drift"):
        generator.load_generation_config(CONFIG_PATH)


def test_public_pilot_cli_has_no_freeze_gate_bypass() -> None:
    forbidden_output = ROOT / ".r2-test-forbidden-pilot"
    with pytest.raises(SystemExit) as excinfo:
        generator.main([
            "pilot",
            "--no-contract-marker-check",
            "--output",
            str(forbidden_output),
        ])
    assert excinfo.value.code == 2
    assert not forbidden_output.exists()


def test_pilot_api_requires_contract_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = ROOT / ".r2-test-missing-marker.txt"
    output = ROOT / ".r2-test-blocked-pilot"
    monkeypatch.setattr(generator, "CONTRACT_FREEZE_MARKER", marker)
    with pytest.raises(generator.GenerationContractError, match="pilot execution is forbidden"):
        generator.run_generation_cohort("pilot", output, "blocked-by-test")
    with pytest.raises(generator.GenerationContractError, match="pilot execution is forbidden"):
        generator.run_pilot(output)
    assert not output.exists()


def test_candidate_pool_generation_is_deterministic(config: dict) -> None:
    small = copy.deepcopy(config)
    small["sampling"]["pools"]["pilot"] = {
        "interior_train": {"candidate_count": 12, "required_quota": 3},
        "wide_valid_train": {"candidate_count": 8, "required_quota": 2},
    }
    first = generator.generate_candidate_pools("pilot", small)
    second = generator.generate_candidate_pools("pilot", small)
    for distribution in ("interior_train", "wide_valid_train"):
        pd.testing.assert_frame_equal(first[distribution], second[distribution])
        assert len(first[distribution]) == small["sampling"]["pools"]["pilot"][distribution]["candidate_count"]
        assert first[distribution]["candidate_id"].tolist() == list(range(len(first[distribution])))
        assert "split" not in first[distribution].columns
        assert "reviewed_sampler_diagnostic_split" in first[distribution].columns


def _candidate_frame(
    distribution: str,
    accepted_flags: list[bool],
) -> pd.DataFrame:
    distribution_offset = 100 if distribution == "wide_valid_train" else 0
    rows = []
    for candidate_id, accepted in enumerate(accepted_flags):
        vector = {
            name: float(distribution_offset + candidate_id + 1)
            for name in PARAMETER_NAMES
        }
        rows.append({
            "candidate_id": candidate_id,
            "distribution": distribution,
            "accepted": accepted,
            "reviewed_sampler_diagnostic_split": "historical_diagnostic_only",
            **vector,
        })
    return pd.DataFrame(rows)


def test_selection_is_ascending_exact_and_never_replaces_failures(
    config: dict,
) -> None:
    local = copy.deepcopy(config)
    local = copy.deepcopy(local)
    local["sampling"]["pools"]["pilot"] = {
        "interior_train": {"candidate_count": 4, "required_quota": 3},
        "wide_valid_train": {"candidate_count": 4, "required_quota": 2},
    }
    local["quotas"]["development_pilot_not_final_research_dataset"]["splits"] = {
        "train": {"total": 3, "interior_train": 2, "wide_valid_train": 1},
        "validation": {"total": 2, "interior_train": 1, "wide_valid_train": 1},
        "test": {"total": 0, "interior_train": 0, "wide_valid_train": 0},
    }
    pools = {
        "interior_train": _candidate_frame("interior_train", [True, True, False, True]),
        "wide_valid_train": _candidate_frame("wide_valid_train", [False, True, True, True]),
    }
    selected, retained = generator.select_accepted_candidates(pools, "pilot", local)
    assert selected["distribution"].tolist() == [
        "interior_train",
        "interior_train",
        "interior_train",
        "wide_valid_train",
        "wide_valid_train",
    ]
    assert selected["candidate_id"].tolist() == [0, 1, 3, 1, 2]
    assert selected["split"].tolist() == [
        "train",
        "train",
        "validation",
        "train",
        "validation",
    ]
    assert len(selected) == sum(
        part["required_quota"] for part in local["sampling"]["pools"]["pilot"].values()
    ) == 5
    assert retained.groupby("distribution")["selected_for_clean_core"].sum().to_dict() == {
        "interior_train": 3,
        "wide_valid_train": 2,
    }
    vector_tuples = list(selected[PARAMETER_NAMES].itertuples(index=False, name=None))
    assert len(vector_tuples) == len(set(vector_tuples))


def test_insufficient_fixed_pool_hard_fails_without_refill(config: dict) -> None:
    local = copy.deepcopy(config)
    local["sampling"]["pools"]["pilot"] = {
        "interior_train": {"candidate_count": 3, "required_quota": 2},
        "wide_valid_train": {"candidate_count": 0, "required_quota": 0},
    }
    local["quotas"]["development_pilot_not_final_research_dataset"]["splits"] = {
        split: {"total": 0, "interior_train": 0, "wide_valid_train": 0}
        for split in ("train", "validation", "test")
    }
    local["quotas"]["development_pilot_not_final_research_dataset"]["splits"]["train"] = {
        "total": 2, "interior_train": 2, "wide_valid_train": 0,
    }
    pools = {
        "interior_train": _candidate_frame("interior_train", [True, False, False]),
        "wide_valid_train": _candidate_frame("wide_valid_train", []),
    }
    with pytest.raises(generator.CandidatePoolInsufficientError, match="refill/reseed"):
        generator.select_accepted_candidates(pools, "pilot", local)


def test_insufficient_pool_failure_retains_complete_candidate_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pools = {
        "interior_train": _candidate_frame("interior_train", [True, False]),
        "wide_valid_train": _candidate_frame("wide_valid_train", [False]),
    }
    written: dict[str, object] = {}
    monkeypatch.setattr(generator, "_write_csv", lambda frame, path: "candidates-hash")
    monkeypatch.setattr(generator, "_write_jsonl", lambda records, path: "rejections-hash")

    def record_json(path: Path, payload: dict) -> None:
        written["path"] = path
        written["payload"] = payload

    monkeypatch.setattr(generator, "write_json", record_json)
    generator._retain_insufficient_pool_evidence(
        Path("controlled/retention"), pools, "pilot"
    )
    assert written["path"] == Path("controlled/retention/manifest.json")
    payload = written["payload"]
    assert payload["status"] == "FAILED_INSUFFICIENT_FIXED_POOL_RETAINED_CANDIDATES"
    assert payload["refill_or_reseed_used"] is False
    assert payload["candidates_csv_sha256"] == "candidates-hash"
    assert payload["rejections_jsonl_sha256"] == "rejections-hash"
    assert payload["sufficiency"]["interior_train"]["candidate_count"] == 2
    assert payload["sufficiency"]["interior_train"]["accepted_count"] == 1
    assert payload["sufficiency"]["interior_train"]["required_quota"] == 200
    assert payload["sufficiency"]["wide_valid_train"]["candidate_count"] == 1
    assert payload["sufficiency"]["wide_valid_train"]["accepted_count"] == 0
    assert payload["sufficiency"]["wide_valid_train"]["required_quota"] == 40


def _surface_for_conditioning(conditioning, provenance: dict):
    dte1 = int(provenance["rank1_dte_days"])
    dte2 = int(provenance["rank2_dte_days"])
    rate = float(provenance["rate"])
    carry = float(provenance["carry"])
    return surface_from_vectors(
        prices=[1.0] * 20,
        mask=[True] * 20,
        maturities=[dte1 / 365.0] * 5 + [dte2 / 365.0] * 5 + [dte1 / 365.0] * 5 + [dte2 / 365.0] * 5,
        rates=[rate] * 20,
        carries=[carry] * 20,
        spot=100.0,
        surface_id=f"conditioning_{provenance['generation_index']}",
        source="unit_test_only_not_evidence",
        metadata={
            "dte": [dte1, dte2],
            "unit_test_only": True,
        },
    )


def test_conditioning_mapping_support_and_validation(config: dict) -> None:
    combination_count = int(config["conditioning"]["lattice"]["combination_count"])
    indices = [0, 1, 100, combination_count - 1]
    surfaces = []
    records = []
    for generation_index in indices:
        conditioning, provenance = generator.build_conditioning(
            generation_index, "pilot", config
        )
        record = {
            **provenance,
            "lattice_index": provenance["lattice_index"],
        }
        surfaces.append(_surface_for_conditioning(conditioning, provenance))
        records.append(record)
        assert conditioning.dte[1] > conditioning.dte[0]
        assert conditioning.rates == (conditioning.rates[0], conditioning.rates[0])
        assert conditioning.carries == (conditioning.carries[0], conditioning.carries[0])
        assert abs(record["carry"] - (record["rate"] + record["carry_offset"])) < 1e-15
    generator.validate_conditioning_support(
        surfaces,
        records,
        combination_count,
        int(config["conditioning"]["seeds"]["pilot"]),
    )
    expected_first = (997 % combination_count)
    assert records[1]["lattice_index"] == expected_first
    assert records[-1]["lattice_index"] == ((combination_count - 1) * 997) % combination_count


def test_selected_surface_failure_retains_identity_conditioning_and_error(
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import r2_representation

    def fail_every_surface(*args, **kwargs):
        raise RuntimeError("controlled pricing failure")

    monkeypatch.setattr(r2_representation, "build_synthetic_surface", fail_every_surface)
    frame = _candidate_frame("interior_train", [True, True])
    frame["candidate_key"] = [f"interior_train_{value:06d}" for value in frame["candidate_id"]]
    frame["split"] = ["train", "validation"]
    hashes = {
        "config": "0" * 64,
        "pricer": "1" * 64,
        "generator_source": "2" * 64,
        "reviewed_sampler_source": "3" * 64,
        "r2_synthetic_interface": "4" * 64,
        "generator_version": "test",
        "parameter_sampler_seeds": {"interior_train": 20260822},
    }
    surfaces, conditioning_rows, failures, sanity_rows = generator.generate_selected_surfaces(
        frame, "pilot", config, hashes
    )
    assert surfaces == []
    assert conditioning_rows == []
    assert sanity_rows == []
    assert len(failures) == 2
    for index, failure in enumerate(failures):
        assert failure["candidate_id"] == index
        assert failure["split"] == ["train", "validation"][index]
        assert failure["error_type"] == "RuntimeError"
        assert failure["error"] == "controlled pricing failure"
        assert failure["parameter_vector_hash"]
        assert failure["conditioning"]["spot"] == 100.0
        assert failure["rank2_dte_days"] > failure["rank1_dte_days"]
        assert failure["real_market_inputs_used"] is False


def test_final_readiness_requires_verified_authoritative_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot_root = ROOT / ".r2-test-missing-authoritative-pilot"
    readiness_output = ROOT / ".r2-test-blocked-readiness"
    monkeypatch.setattr(generator, "PILOT_OUTPUT", pilot_root)
    with pytest.raises(generator.GenerationContractError, match="verified pilot"):
        generator.run_final_readiness(readiness_output)
    assert not readiness_output.exists()


def test_final_readiness_source_contains_no_surface_pricing_path() -> None:
    text = Path(generator.__file__).read_text(encoding="utf-8")
    start = text.index("def run_final_readiness(")
    end = text.index("\ndef main(", start)
    body = text[start:end]
    assert "_build_generation_cohort" not in body
    assert "price_double_heston_surface" not in body
    assert '"surfaces_generated": False' in body
    assert '"pricing_performed": False' in body


def test_internal_builder_physically_prohibits_final_surface_generation() -> None:
    output = ROOT / ".r2-test-forbidden-final-generation"
    with pytest.raises(
        generator.GenerationContractError,
        match="final 10k pricing is separately gated",
    ):
        generator._build_generation_cohort(
            "final",
            output,
            "forbidden-by-test",
        )
    assert not output.exists()


def test_manifest_artifact_hashes_match_files_on_disk(
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    small = copy.deepcopy(config)
    small["sampling"]["pools"]["pilot"] = {
        "interior_train": {"candidate_count": 40, "required_quota": 3},
        "wide_valid_train": {"candidate_count": 24, "required_quota": 2},
    }
    small["quotas"]["development_pilot_not_final_research_dataset"] = {
        "total_surfaces": 5,
        "distributions": {"interior_train": 3, "wide_valid_train": 2},
        "splits": {
            "train": {"total": 3, "interior_train": 2, "wide_valid_train": 1},
            "validation": {"total": 1, "interior_train": 1, "wide_valid_train": 0},
            "test": {"total": 1, "interior_train": 0, "wide_valid_train": 1},
        },
    }
    monkeypatch.setattr(generator, "load_generation_config", lambda path=None: small)
    output = tmp_path / "pilot"
    _, manifest = generator._build_generation_cohort(
        "pilot", output, "hash-consistency-test"
    )
    artifact_files = {
        "surfaces_jsonl_sha256": "surfaces.jsonl",
        "selected_parameters_csv_sha256": "selected_parameters.csv",
        "candidates_csv_sha256": "candidates.csv",
        "rejections_jsonl_sha256": "rejections.jsonl",
        "conditioning_jsonl_sha256": "conditioning.jsonl",
        "numerical_sanity_jsonl_sha256": "numerical_sanity.jsonl",
        "integrity_report_json_sha256": "integrity_report.json",
    }
    for hash_field, filename in artifact_files.items():
        artifact = output / filename
        assert artifact.is_file()
        assert generator.sha256_file(artifact) == manifest["provenance_hashes"][hash_field], (
            f"manifest hash for {filename} does not match the file on disk"
        )
    json.loads((output / "integrity_report.json").read_text(encoding="utf-8"))


def _gate_payload(index: int, distribution: str, split: str) -> dict:
    values = {name: 0.5 for name in PARAMETER_NAMES}
    values["v0_slow"] = 0.01 + index * 1e-6
    moneyness = [-0.10, -0.05, 0.0, 0.05, 0.10]
    slot_keys = [
        [rank, level, option]
        for option in ("call", "put")
        for rank in (1, 2)
        for level in moneyness
    ]
    return {
        "representation_name": "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE",
        "representation_version": "1.0",
        "slot_keys": slot_keys,
        "prices": [1.0 + 0.01 * position for position in range(20)],
        "mask": [True] * 20,
        "maturities": [7.0 / 365.0] * 10 + [14.0 / 365.0] * 10,
        "rates": [0.02] * 20,
        "carries": [0.01] * 20,
        "spot": 100.0,
        "surface_id": f"R2_PILOT_fake_{index:06d}",
        "source": "synthetic",
        "metadata": {
            "parameters_canonical_order": values,
            "user_metadata": {"distribution": distribution, "split": split},
        },
    }


def _write_gate_evidence(root: Path, payloads: list[dict]) -> None:
    root.mkdir(parents=True)
    surfaces_payload = b"".join(
        generator.deterministic_json_bytes(payload) for payload in payloads
    )
    (root / "surfaces.jsonl").write_bytes(surfaces_payload)
    hashes = {"surfaces_jsonl_sha256": generator.sha256_bytes(surfaces_payload)}
    artifacts = (
        ("selected_parameters_csv_sha256", "selected_parameters.csv", None),
        ("candidates_csv_sha256", "candidates.csv", None),
        ("rejections_jsonl_sha256", "rejections.jsonl", []),
        ("conditioning_jsonl_sha256", "conditioning.jsonl", []),
        ("numerical_sanity_jsonl_sha256", "numerical_sanity.jsonl", []),
    )
    for field, filename, records in artifacts:
        if records is None:
            frame = pd.DataFrame({"placeholder": [1]})
            hashes[field] = generator._write_csv(frame, root / filename)
        else:
            hashes[field] = generator._write_jsonl(records, root / filename)
    integrity = {
        "representation_validated": True,
        "all_masks_true": True,
        "all_slot_counts_20": True,
        "unique_surface_ids": True,
        "unique_parameter_vectors": True,
        "serialization_round_trip": True,
    }
    integrity_bytes = generator.deterministic_json_bytes(integrity)
    (root / "integrity_report.json").write_bytes(integrity_bytes)
    hashes["integrity_report_json_sha256"] = generator.sha256_bytes(integrity_bytes)
    return hashes


def test_readiness_gate_accepts_well_formed_pilot_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allocation = [
        ("interior_train", "train", 150),
        ("interior_train", "validation", 25),
        ("interior_train", "test", 25),
        ("wide_valid_train", "train", 30),
        ("wide_valid_train", "validation", 5),
        ("wide_valid_train", "test", 5),
    ]
    payloads: list[dict] = []
    for distribution, split, count in allocation:
        for _ in range(count):
            payloads.append(_gate_payload(len(payloads), distribution, split))
    assert len(payloads) == 240

    primary = tmp_path / "pilot"
    replay = tmp_path / "pilot_replay"
    primary_hashes = _write_gate_evidence(primary, payloads)
    replay_hashes = _write_gate_evidence(replay, payloads)
    hash_fields = [
        "surfaces_jsonl_sha256",
        "selected_parameters_csv_sha256",
        "candidates_csv_sha256",
        "rejections_jsonl_sha256",
        "conditioning_jsonl_sha256",
        "numerical_sanity_jsonl_sha256",
        "integrity_report_json_sha256",
    ]
    primary_manifest = {
        "cohort": "pilot",
        "surface_count": 240,
        "replay_status": "VERIFIED_IDENTICAL",
        "replay_report": {
            "identical": True,
            "hash_comparisons": {field: True for field in hash_fields},
        },
        "numerical_sanity": {"all_passed": True},
        "provenance_hashes": primary_hashes,
    }
    replay_manifest = {
        "cohort": "pilot",
        "surface_count": 240,
        "replay_status": "PENDING",
        "provenance_hashes": replay_hashes,
    }
    generator.write_json(primary / "manifest.json", primary_manifest)
    generator.write_json(replay / "manifest.json", replay_manifest)
    monkeypatch.setattr(generator, "PILOT_OUTPUT", primary)
    evidence = generator._verified_pilot_evidence()
    assert evidence["primary_manifest"]["surface_count"] == 240

    duplicated = list(payloads)
    duplicated[1] = {**duplicated[1], "metadata": {**duplicated[1]["metadata"]}}
    duplicated[1]["metadata"]["parameters_canonical_order"] = dict(
        payloads[0]["metadata"]["parameters_canonical_order"]
    )
    tampered = tmp_path / "tampered"
    tampered_hashes = _write_gate_evidence(tampered, duplicated)
    tampered_manifest = {
        **primary_manifest,
        "provenance_hashes": tampered_hashes,
    }
    generator.write_json(tampered / "manifest.json", tampered_manifest)
    tampered_replay = tmp_path / "tampered_replay"
    tampered_replay_hashes = _write_gate_evidence(tampered_replay, duplicated)
    generator.write_json(
        tampered_replay / "manifest.json",
        {**replay_manifest, "provenance_hashes": tampered_replay_hashes},
    )
    monkeypatch.setattr(generator, "PILOT_OUTPUT", tampered)
    with pytest.raises(generator.GenerationContractError, match="duplicate identity"):
        generator._verified_pilot_evidence()


def test_final_readiness_insufficient_pool_fails_closed_with_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The insufficient FINAL readiness path itself must fail closed with
    complete retention — not crash before it can preserve evidence."""
    from src import r2_representation

    pool_calls = {"count": 0}

    def _pool_frame(distribution: str, accepted_count: int, rejected_count: int) -> pd.DataFrame:
        rows = []
        for i in range(accepted_count + rejected_count):
            row = {
                "candidate_id": i,
                "distribution": distribution,
                "accepted": i < accepted_count,
                "rejection_reasons": "" if i < accepted_count else "ordinary_training_margin_policy",
            }
            row.update({name: 0.5 for name in PARAMETER_NAMES})
            rows.append(row)
        return pd.DataFrame(rows)

    interior_frame = _pool_frame("interior_train", 8_334, 10)
    wide_frame = _pool_frame("wide_valid_train", 100, 100)

    def fake_pools(cohort: str, config: dict) -> dict:
        assert cohort == "final"
        pool_calls["count"] += 1
        return {"interior_train": interior_frame, "wide_valid_train": wide_frame}

    def fail_pricing(*args, **kwargs):
        raise AssertionError("final readiness must not price any surface")

    monkeypatch.setattr(generator, "_verified_pilot_evidence", lambda: {"ok": True})
    monkeypatch.setattr(generator, "generate_candidate_pools", fake_pools)
    monkeypatch.setattr(r2_representation, "build_synthetic_surface", fail_pricing)

    output = tmp_path / "readiness"
    with pytest.raises(generator.CandidatePoolInsufficientError) as excinfo:
        generator.run_final_readiness(output)
    assert "rejections were retained" in str(excinfo.value)

    # Exactly one fixed-pool pass: no refill, reseed, or retry loop.
    assert pool_calls["count"] == 1

    # Deterministic failure location derived from output_directory.
    failure_output = Path(str(output) + "_failed_insufficient")
    assert failure_output.is_dir()
    assert sorted(path.name for path in failure_output.iterdir()) == [
        "all_final_candidates.csv",
        "readiness_manifest.json",
        "rejections.jsonl",
    ]
    # No success output and no surface dataset anywhere.
    assert not output.exists()
    assert not list(failure_output.glob("surfaces*"))

    manifest = json.loads((failure_output / "readiness_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED_INSUFFICIENT_FIXED_POOL_RETAINED_REJECTIONS"
    assert manifest["all_sufficient"] is False
    assert manifest["selected_panel_count"] == 0
    assert manifest["surfaces_generated"] is False
    assert manifest["pricing_performed"] is False
    assert manifest["final_10k_generated"] is False
    assert manifest["training_started"] is False
    assert manifest["sufficiency"]["interior_train"] == {
        "pool_count": 8_344,
        "accepted_count": 8_334,
        "required_quota": 8_334,
        "sufficient": True,
    }
    assert manifest["sufficiency"]["wide_valid_train"] == {
        "pool_count": 200,
        "accepted_count": 100,
        "required_quota": 1_666,
        "sufficient": False,
    }

    # Complete fixed candidate/rejection evidence retained and hash-sealed.
    candidates = pd.read_csv(failure_output / "all_final_candidates.csv")
    assert len(candidates) == 8_344 + 200
    rejection_lines = [
        line for line in (failure_output / "rejections.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rejection_lines) == 10 + 100
    assert all(json.loads(line)["accepted"] is False for line in rejection_lines)
    assert generator.sha256_file(failure_output / "all_final_candidates.csv") == manifest["all_candidates_csv_sha256"]
    assert generator.sha256_file(failure_output / "rejections.jsonl") == manifest["rejections_jsonl_sha256"]
