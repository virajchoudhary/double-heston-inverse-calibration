"""Focused tests for the explicit final R2 10k generation gate.

Proves (spec section "TEST THE FINAL-GENERATION GATE"):
1. the generic/internal surface builder still rejects final generation;
2. the final-generation CLI/API rejects without the authorization marker;
3. the authorization marker must contain the correct frozen identities;
4. a wrong parameter-panel hash rejects;
5. a wrong config hash rejects;
6. a wrong pricer/R2 identity rejects;
7. an incorrect total count rejects;
8. duplicate vectors reject;
9. split mismatch rejects;
10. an existing output directory rejects;
11. no final pricing occurs during preflight;
12. only the explicit final-generation pathway can price the final cohort;
13. no training is started by final generation;
14. no real-market data is accessed by final generation;
15. selected pricing failures are retained and never replaced/refilled;
16. no post-hoc reseed or candidate selection exists.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src import r2_final_generation as rfg
from src import r2_synthetic_generation as frozen
from src.constants import PARAMETER_NAMES

ROOT = Path(__file__).resolve().parents[1]
MODULE_SOURCE = Path(rfg.__file__).read_text(encoding="utf-8")


def _fake_authorization() -> dict:
    return {
        "authorized": True,
        "marker_path": "test-marker",
        "marker_sha256": "0" * 64,
        "canonical_main_base_git_sha": "0" * 40,
        "authorization_commit_sha": "1" * 40,
        "identities": {},
    }


def _marker_text(**overrides: str) -> str:
    hashes = rfg.scientific_dependency_hashes()
    fields = {
        "canonical_main_base_git_sha": "75ad4d014f19cd645722c33f9c244b57759121fc",
        "frozen_contract_reference": "configs/r2_synthetic_generation_FINAL.yaml",
        "frozen_parameter_panel_sha256": hashes["panel"],
        "production_pricer_sha256": hashes["pricer"],
        "r2_synthetic_interface_sha256": hashes["r2_synthetic_interface"],
        "generation_config_sha256": hashes["config"],
        "generator_source_sha256": hashes["generator_source"],
        "final_generation_module_sha256": hashes["final_generation_module"],
        "final_quota_surfaces": "10000",
        "noise_level": "0.0",
        "real_market_inputs": "NONE",
        "training_authorization": "NONE",
        "g8_authorization": "NONE",
    }
    fields.update(overrides)
    lines = [
        "FINAL_R2_10K_GENERATION_AUTHORIZED",
        "",
        f"statement: {rfg.NO_PRIOR_OUTPUT_STATEMENT}",
        "",
    ]
    lines.extend(f"{key}: {value}" for key, value in fields.items())
    return "\n".join(lines) + "\n"


@pytest.fixture()
def committed_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """A marker whose repository-state gates all pass."""
    marker = tmp_path / "R2_FINAL_10K_GENERATION_AUTHORIZED.txt"
    marker.write_text(_marker_text(), encoding="utf-8")
    monkeypatch.setattr(rfg, "_marker_committed_at_head", lambda path=None: True)
    monkeypatch.setattr(rfg, "_base_is_ancestor", lambda sha: True)
    monkeypatch.setattr(rfg, "_head_on_remote", lambda sha: True)
    monkeypatch.setattr(rfg, "_current_head", lambda: "1" * 40)
    return marker


@pytest.fixture(scope="module")
def real_panel_slice() -> pd.DataFrame:
    panel = rfg.load_final_panel()
    return panel.iloc[[0, 1, 2, 8334, 8335, 9997, 9998, 9999]].copy()


@pytest.fixture(scope="module")
def real_full_panel() -> pd.DataFrame:
    return rfg.load_final_panel()


# 1. -------------------------------------------------------------------------

def test_internal_surface_builder_still_rejects_final() -> None:
    output = ROOT / ".r2-test-forbidden-final-internal"
    with pytest.raises(
        frozen.GenerationContractError, match="final 10k pricing is separately gated"
    ):
        frozen._build_generation_cohort("final", output, "forbidden-by-test")
    assert not output.exists()


def test_frozen_generator_config_gate_remains_closed() -> None:
    config = frozen.load_generation_config()
    assert (
        config["execution_gates"]["final_10k_generation_command"]
        == "NOT_AUTHORIZED_IN_THIS_MILESTONE"
    )


# 2. -------------------------------------------------------------------------

def test_generate_final_rejects_without_authorization_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rfg, "AUTHORIZATION_MARKER", tmp_path / "absent-marker.txt")
    output = tmp_path / "final"
    with pytest.raises(
        rfg.FinalGenerationAuthorizationError, match="NOT AUTHORIZED"
    ):
        rfg.run_final_generation(output)
    assert not output.exists()


def test_generate_final_rejects_uncommitted_marker(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rfg, "_marker_committed_at_head", lambda path=None: False
    )
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="not committed at HEAD"):
        rfg.verify_authorization(committed_marker)


def test_generate_final_rejects_when_base_not_ancestor(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rfg, "_base_is_ancestor", lambda sha: False)
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="ancestor"):
        rfg.verify_authorization(committed_marker)


def test_generate_final_rejects_when_commit_not_on_remote(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rfg, "_head_on_remote", lambda sha: False)
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="remote"):
        rfg.verify_authorization(committed_marker)


# 3. -------------------------------------------------------------------------
# (also 4, 5, 6: wrong identities must reject)


def _verify_with_tampered_marker(
    committed_marker: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: str,
) -> None:
    committed_marker.write_text(_marker_text(**overrides), encoding="utf-8")
    rfg.verify_authorization(committed_marker)


def test_marker_with_wrong_panel_hash_rejects(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="frozen_parameter_panel_sha256"):
        _verify_with_tampered_marker(
            committed_marker, monkeypatch, frozen_parameter_panel_sha256="f" * 64
        )


def test_marker_with_wrong_config_hash_rejects(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="generation_config_sha256"):
        _verify_with_tampered_marker(
            committed_marker, monkeypatch, generation_config_sha256="c" * 64
        )


def test_marker_with_wrong_pricer_hash_rejects(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="production_pricer_sha256"):
        _verify_with_tampered_marker(
            committed_marker, monkeypatch, production_pricer_sha256="a" * 64
        )


def test_marker_with_wrong_r2_interface_hash_rejects(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="r2_synthetic_interface_sha256"):
        _verify_with_tampered_marker(
            committed_marker, monkeypatch, r2_synthetic_interface_sha256="b" * 64
        )


def test_marker_with_wrong_module_hash_rejects_postauthorization_edits(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="final_generation_module_sha256"):
        _verify_with_tampered_marker(
            committed_marker, monkeypatch, final_generation_module_sha256="d" * 64
        )


def test_marker_with_wrong_quota_or_noise_or_authorizations_rejects(
    committed_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for field, value in (
        ("final_quota_surfaces", "9999"),
        ("noise_level", "0.01"),
        ("real_market_inputs", "ARCHIVE2"),
        ("training_authorization", "GRANTED"),
        ("g8_authorization", "GRANTED"),
    ):
        with pytest.raises(rfg.FinalGenerationAuthorizationError):
            _verify_with_tampered_marker(
                committed_marker, monkeypatch, **{field: value}
            )


def test_marker_missing_no_prior_output_statement_rejects(tmp_path: Path) -> None:
    text = _marker_text().replace(
        f"statement: {rfg.NO_PRIOR_OUTPUT_STATEMENT}", "statement: trimmed"
    )
    marker = tmp_path / "marker.txt"
    marker.write_text(text, encoding="utf-8")
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="required statement"):
        rfg.parse_authorization_marker(marker)


def test_marker_missing_required_field_rejects(tmp_path: Path) -> None:
    text = _marker_text()
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith("production_pricer_sha256")
    )
    marker = tmp_path / "marker.txt"
    marker.write_text(text, encoding="utf-8")
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="missing fields"):
        rfg.parse_authorization_marker(marker)


# 7. -------------------------------------------------------------------------

def test_preflight_rejects_incorrect_total_count(
    real_panel_slice: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: real_panel_slice)
    with pytest.raises(rfg.FinalDatasetValidationError, match="panel_count_10000"):
        rfg.run_preflight()


# 8. -------------------------------------------------------------------------

def test_preflight_rejects_duplicate_parameter_vectors(
    real_full_panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicated = real_full_panel.copy()
    # Replace the last row's vector with row 0's vector: count stays 10,000
    # but a duplicate parameter vector now exists.
    duplicated.loc[duplicated.index[-1], list(PARAMETER_NAMES)] = real_full_panel.loc[
        real_full_panel.index[0], list(PARAMETER_NAMES)
    ]
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: duplicated)
    with pytest.raises(rfg.FinalDatasetValidationError, match="unique_parameter_vectors"):
        rfg.run_preflight()


# 9. -------------------------------------------------------------------------

def test_preflight_rejects_split_mismatch(
    real_full_panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = real_full_panel.copy()
    tampered.iloc[0, tampered.columns.get_loc("split")] = "test"
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: tampered)
    with pytest.raises(rfg.FinalDatasetValidationError, match="split_quotas_exact"):
        rfg.run_preflight()


def test_preflight_rejects_panel_hash_column_inconsistency(
    real_full_panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = real_full_panel.copy()
    tampered.iloc[0, tampered.columns.get_loc("parameter_vector_hash")] = "0" * 64
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: tampered)
    with pytest.raises(rfg.FinalDatasetValidationError, match="panel_hash_column_consistent"):
        rfg.run_preflight()


# 10. ------------------------------------------------------------------------

def test_generation_refuses_existing_output_directory(
    tmp_path: Path,
    real_panel_slice: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "final"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(rfg, "verify_authorization", lambda marker=None: _fake_authorization())
    monkeypatch.setattr(rfg, "run_preflight", lambda *args, **kwargs: {"passed": True})
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="refusing to overwrite"):
        rfg.run_final_generation(output)
    assert (output / "stale.txt").read_text(encoding="utf-8") == "stale"


# 11. ------------------------------------------------------------------------

def test_preflight_performs_no_price_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import double_heston
    from src import r2_representation

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must never price a surface")

    monkeypatch.setattr(double_heston, "price_double_heston_surface", forbidden)
    monkeypatch.setattr(r2_representation, "build_synthetic_surface", forbidden)
    # Post-generation the sealed output exists, so the no-price-call proof uses
    # the post-generation phase; pre-generation the default behaves the same.
    report = rfg.run_preflight(require_final_output_absent=False)
    assert report["passed"] is True
    assert report["no_price_calls"] is True


# 12. ------------------------------------------------------------------------

def test_only_explicit_final_pathway_can_price_final_cohort() -> None:
    # The frozen generator's public cohort runner and internal builder both
    # refuse the final cohort; readiness never prices; only this module's
    # authorized generate-final pathway can.
    output = ROOT / ".r2-test-forbidden-final-public"
    with pytest.raises(
        frozen.GenerationContractError, match="final 10k pricing is separately gated"
    ):
        frozen.run_generation_cohort("final", output, "forbidden-by-test")
    assert not output.exists()
    readiness_source = Path(frozen.__file__).read_text(encoding="utf-8")
    start = readiness_source.index("def run_final_readiness(")
    end = readiness_source.index("\ndef main(", start)
    assert "price_double_heston_surface" not in readiness_source[start:end]
    assert "_build_generation_cohort" not in readiness_source[start:end]
    # And the explicit pathway itself is gated behind verify_authorization,
    # which is the first call inside run_final_generation.


# 13. ------------------------------------------------------------------------

def test_final_generation_starts_no_training(
    tmp_path: Path,
    real_panel_slice: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rfg, "verify_authorization", lambda marker=None: _fake_authorization())
    monkeypatch.setattr(rfg, "run_preflight", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: real_panel_slice)
    monkeypatch.setattr(
        rfg,
        "validate_final_dataset",
        lambda output, panel=None, expected_total=rfg.FINAL_TOTAL_SURFACES: {
            "validated": True,
            "checks": {},
            "surface_count": len(panel),
        },
    )
    output, manifest = rfg.run_final_generation(tmp_path / "final")
    assert manifest["training_started"] is False
    assert manifest["g8_started"] is False
    assert manifest["noise_level"] == 0.0
    for forbidden in ("train.py", "train_pinn", "torch", "fit(", "optimizer"):
        assert forbidden not in MODULE_SOURCE
    config = frozen.load_generation_config()
    assert config["execution_gates"]["training_commands_in_this_milestone"] == "NONE"


# 14. ------------------------------------------------------------------------

def test_final_generation_accesses_no_real_market_data(
    tmp_path: Path,
    real_panel_slice: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rfg, "verify_authorization", lambda marker=None: _fake_authorization())
    monkeypatch.setattr(rfg, "run_preflight", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: real_panel_slice)
    monkeypatch.setattr(
        rfg,
        "validate_final_dataset",
        lambda output, panel=None, expected_total=rfg.FINAL_TOTAL_SURFACES: {
            "validated": True,
            "checks": {},
            "surface_count": len(panel),
        },
    )
    output, manifest = rfg.run_final_generation(tmp_path / "final")
    assert manifest["real_market_inputs_used"] is False
    for line in (output / "surfaces.jsonl").read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        assert payload["source"] == rfg.SOURCE_SYNTHETIC
        assert payload["metadata"]["synthetic"] is True
        assert payload["metadata"]["user_metadata"]["real_market_inputs_used"] is False
    for forbidden in ("build_real_surface", "market_data", "real_finetune", "nse_stage_a"):
        assert forbidden not in MODULE_SOURCE


# 15. ------------------------------------------------------------------------

def test_pricing_failures_are_retained_and_never_replaced(
    tmp_path: Path,
    real_panel_slice: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import r2_representation

    panel_calls = {"count": 0}

    def counted_panel_load(path=None):
        panel_calls["count"] += 1
        return real_panel_slice

    def fail_every_surface(*args, **kwargs):
        raise RuntimeError("controlled final pricing failure")

    monkeypatch.setattr(rfg, "verify_authorization", lambda marker=None: _fake_authorization())
    monkeypatch.setattr(rfg, "run_preflight", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(rfg, "load_final_panel", counted_panel_load)
    monkeypatch.setattr(r2_representation, "build_synthetic_surface", fail_every_surface)

    output = tmp_path / "final"
    with pytest.raises(rfg.FinalDatasetValidationError, match="FAILED CLOSED"):
        rfg.run_final_generation(output)

    # Exactly one pass over the fixed panel: no refill/reseed/retry loop.
    assert panel_calls["count"] == 1
    assert not (output / "surfaces.jsonl").exists()
    failures = [
        json.loads(line)
        for line in (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(failures) == len(real_panel_slice)
    for failure in failures:
        assert failure["dataset_status"] == "RETAINED_FINAL_GENERATION_FAILURE"
        assert failure["error_type"] == "RuntimeError"
        assert failure["error"] == "controlled final pricing failure"
        assert failure["candidate_key"]
        assert failure["split"] in ("train", "validation", "test")
        assert set(failure["parameters"]) == set(PARAMETER_NAMES)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED_CLOSED_RETAINED_FAILURES"
    assert manifest["failure_count"] == len(real_panel_slice)
    assert manifest["replacement_or_refill_used"] is False


# 16. ------------------------------------------------------------------------

def test_no_post_hoc_reseed_or_candidate_selection_exists() -> None:
    for forbidden in (
        "generate_candidate_pools(",
        "sample_distribution(",
        "load_reviewed_config(",
        "np.random",
        "random.seed",
    ):
        assert forbidden not in MODULE_SOURCE, forbidden


def test_predeclared_replay_subset_is_rule_based_only(
    real_panel_slice: pd.DataFrame,
) -> None:
    first = rfg.predeclared_replay_indices(real_panel_slice)
    structurally_identical = real_panel_slice.copy()
    structurally_identical[PARAMETER_NAMES] = 0.5
    second = rfg.predeclared_replay_indices(structurally_identical)
    assert first == second
    assert first == sorted(set(first))
    assert 0 in first
    assert len(first) >= 1


# Additional gate-integrity checks -------------------------------------------


def test_authorize_refuses_to_overwrite_existing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "marker.txt"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="overwrite"):
        rfg.write_authorization_marker(marker)


def test_authorize_writes_marker_only_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rfg,
        "run_preflight",
        lambda: (_ for _ in ()).throw(rfg.FinalDatasetValidationError("preflight failed: x")),
    )
    marker = tmp_path / "marker.txt"
    with pytest.raises(rfg.FinalDatasetValidationError):
        rfg.write_authorization_marker(marker)
    assert not marker.exists()


def test_cli_has_no_bypass_flags() -> None:
    for forbidden in ("--force", "--unsafe", "--yes", "--allow"):
        with pytest.raises(SystemExit) as excinfo:
            rfg.main(["generate-final", forbidden])
        assert excinfo.value.code == 2


def test_generation_output_metadata_carries_required_provenance(
    tmp_path: Path,
    real_panel_slice: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rfg, "verify_authorization", lambda marker=None: _fake_authorization())
    monkeypatch.setattr(rfg, "run_preflight", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: real_panel_slice)
    monkeypatch.setattr(
        rfg,
        "validate_final_dataset",
        lambda output, panel=None, expected_total=rfg.FINAL_TOTAL_SURFACES: {
            "validated": True,
            "checks": {},
            "surface_count": len(panel),
        },
    )
    output, _ = rfg.run_final_generation(tmp_path / "final")
    line = (output / "surfaces.jsonl").read_text(encoding="utf-8").splitlines()[0]
    user_metadata = json.loads(line)["metadata"]["user_metadata"]
    required = {
        "dataset_status",
        "distribution",
        "split",
        "candidate_id",
        "parameter_vector_hash",
        "parameter_sampler_seed",
        "conditioning_seed",
        "conditioning_lattice_index",
        "generation_config_sha256",
        "generator_version",
        "production_pricer_sha256",
    }
    assert required <= set(user_metadata)
    assert user_metadata["dataset_status"] == rfg.FINAL_DATASET_STATUS
    assert user_metadata["parameter_sampler_seed"] in (20260807, 20260808)


def test_replay_subset_comparison_detects_tampering(
    tmp_path: Path,
    real_panel_slice: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rfg, "verify_authorization", lambda marker=None: _fake_authorization())
    monkeypatch.setattr(rfg, "run_preflight", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(rfg, "load_final_panel", lambda path=None: real_panel_slice)
    monkeypatch.setattr(
        rfg,
        "validate_final_dataset",
        lambda output, panel=None, expected_total=rfg.FINAL_TOTAL_SURFACES: {
            "validated": True,
            "checks": {},
            "surface_count": len(panel),
        },
    )
    primary, _ = rfg.run_final_generation(tmp_path / "final")
    monkeypatch.setattr(rfg, "FINAL_OUTPUT", primary)

    replay_dir = tmp_path / "replay"
    _, report = rfg.run_final_replay(
        primary, replay_dir, mode="predeclared-subset"
    )
    assert report["byte_identical_payloads"] is True
    assert report["full_replay_performed"] is False

    lines = (primary / "surfaces.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])
    payload["prices"][0] = payload["prices"][0] + 1.0
    lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    (primary / "surfaces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(rfg.FinalDatasetValidationError, match="NOT byte-identical"):
        rfg.run_final_replay(primary, tmp_path / "replay2", mode="predeclared-subset")


def test_replay_refuses_overwrite_and_unknown_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rfg, "verify_authorization", lambda marker=None: _fake_authorization())
    monkeypatch.setattr(rfg, "run_preflight", lambda *args, **kwargs: {"passed": True})
    existing = tmp_path / "replay"
    existing.mkdir()
    primary = tmp_path / "final"
    primary.mkdir()
    (primary / "surfaces.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(rfg.FinalGenerationAuthorizationError, match="refusing to overwrite"):
        rfg.run_final_replay(primary, existing, mode="full")
    with pytest.raises(rfg.FinalDatasetValidationError, match="mode"):
        rfg.run_final_replay(primary, tmp_path / "replay3", mode="subset-of-my-choosing")
