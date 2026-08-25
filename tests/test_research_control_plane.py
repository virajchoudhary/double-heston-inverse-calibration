from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.research_control import (
    build_execution_matrix,
    build_report,
    build_tomorrow_handoff,
    detect_stale_documents,
    preflight_lane,
    validate_control_inputs,
    validate_graph,
    validate_registry,
    validate_result_intake,
)


def _minimal_registry() -> dict:
    return {
        "schema_version": "1.0",
        "registry_kind": "RESEARCH_EXPERIMENT_REGISTRY",
        "snapshot": {
            "observed_at_utc": "2026-08-25T00:00:00+00:00",
            "repository": "test",
            "audit_base_commit": "a" * 40,
            "control_branch": "control",
            "remote_namespace": "origin",
            "tracked_tree_required_clean": True,
        },
        "experiments": [
            {
                "id": "parent_lane",
                "title": "Parent",
                "classification": "SEALED_RESULT",
                "scientific_status": "COMPLETE",
                "branch": "main",
                "commit": "b" * 40,
                "base_commit": "a" * 40,
                "protocol": "protocol.md",
                "protocol_sha256": "UNKNOWN",
                "config": "config.yaml",
                "config_sha256": "1" * 64,
                "dataset": "data.jsonl",
                "dataset_sha256": "2" * 64,
                "train_population": "one",
                "validation_population": "none",
                "test_population": "one",
                "test_opened": True,
                "seeds": [1],
                "runner": "runner.py",
                "compute_class": "LIGHT_LOCAL_CPU",
                "execution_location": "local",
                "intended_compute_environment": "local",
                "execution_command": "done",
                "dependencies": [],
                "required_artifacts": [{"ref": "main", "path": "evidence.json", "sha256": "UNKNOWN"}],
                "partial_artifacts": [],
                "result_status": "SEALED",
                "test_exposure_state": "OPENED_AFTER_FREEZE",
                "scientifically_legal_next_action": "preserve",
                "forbidden_actions": ["do not rerun"],
                "stop_gate": "immutable",
            },
            {
                "id": "child_lane",
                "title": "Child",
                "classification": "PENDING",
                "scientific_status": "PENDING",
                "branch": "feature",
                "commit": "c" * 40,
                "base_commit": "a" * 40,
                "protocol": "pending.md",
                "protocol_sha256": "NOT_APPLICABLE",
                "config": "pending.yaml",
                "config_sha256": "NOT_APPLICABLE",
                "dataset": "future.jsonl",
                "dataset_sha256": "NOT_YET_GENERATED",
                "train_population": "none",
                "validation_population": "none",
                "test_population": "closed",
                "test_opened": False,
                "seeds": "not defined",
                "runner": "missing.py",
                "compute_class": "LIGHT_LOCAL_CPU",
                "execution_location": "local",
                "intended_compute_environment": "local",
                "execution_command": "forbidden until review",
                "dependencies": ["parent_lane"],
                "required_artifacts": [],
                "partial_artifacts": [],
                "result_status": "PENDING",
                "test_exposure_state": "CLOSED",
                "scientifically_legal_next_action": "review protocol",
                "forbidden_actions": ["do not execute"],
                "stop_gate": "human review",
            },
        ],
        "stale_rules": [
            {
                "id": "old_parent_claim",
                "file": "status.md",
                "pattern": "parent absent",
                "when_complete_experiment": "parent_lane",
                "reason": "old status",
            }
        ],
        "tomorrow_handoff": {
            "remote_snapshot_utc": "2026-08-25T00:00:00+00:00",
            "changed_since_starting_base": True,
            "first_actions": ["verify"],
            "human_review_gates": ["review all science"],
        },
    }


def _minimal_graph() -> dict:
    return {
        "schema_version": "1.0",
        "graph_kind": "RESEARCH_DEPENDENCY_GRAPH",
        "nodes": [
            {
                "id": "parent_lane",
                "depends_on": [],
                "status_source": "parent_lane",
                "runnable_now": False,
                "runnable_reason": "sealed",
                "required_evidence": ["evidence.json"],
            },
            {
                "id": "child_lane",
                "depends_on": ["parent_lane"],
                "status_source": "child_lane",
                "runnable_now": False,
                "runnable_reason": "needs review",
                "required_evidence": [],
            },
        ],
    }


def test_real_control_inputs_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load((root / "configs/research_experiment_registry.yaml").read_text(encoding="utf-8"))
    graph = yaml.safe_load((root / "configs/research_dependency_graph.yaml").read_text(encoding="utf-8"))
    assert validate_registry(registry) == []
    assert validate_graph(graph, registry) == []


def test_registry_rejects_duplicate_ids_and_missing_dependency() -> None:
    registry = _minimal_registry()
    registry["experiments"].append(copy.deepcopy(registry["experiments"][0]))
    errors = validate_registry(registry)
    assert any("duplicate experiment id" in error for error in errors)

    registry = _minimal_registry()
    registry["experiments"][1]["dependencies"] = ["missing"]
    assert any("missing dependency missing" in error for error in validate_registry(registry))


@pytest.mark.parametrize(
    ("field", "value"),
    [("classification", "MAYBE"), ("scientific_status", "DONE"), ("config_sha256", "123"), ("test_opened", "yes")],
)
def test_registry_rejects_invalid_enums_hashes_and_types(field: str, value: object) -> None:
    registry = _minimal_registry()
    registry["experiments"][0][field] = value
    errors = validate_registry(registry)
    assert errors


def test_graph_rejects_missing_dependency_cycle_and_blocked_runnable() -> None:
    graph = _minimal_graph()
    graph["nodes"][1]["depends_on"] = ["missing"]
    assert any("references missing dependency" in error for error in validate_graph(graph, _minimal_registry()))

    graph = _minimal_graph()
    graph["nodes"][0]["depends_on"] = ["child_lane"]
    assert any("dependency cycle" in error for error in validate_graph(graph, _minimal_registry()))

    graph = _minimal_graph()
    registry = _minimal_registry()
    registry["experiments"][1]["classification"] = "BLOCKED"
    graph["nodes"][1]["runnable_now"] = True
    assert any("blocked lane" in error and "is marked runnable" in error for error in validate_graph(graph, registry))


def test_graph_rejects_complete_child_with_incomplete_parent() -> None:
    registry = _minimal_registry()
    graph = _minimal_graph()
    registry["experiments"][0]["scientific_status"] = "PARTIAL"
    registry["experiments"][0]["classification"] = "PARTIAL_CHECKPOINT"
    registry["experiments"][0]["partial_artifacts"] = [{"ref": "main", "path": "partial.json", "sha256": "UNKNOWN"}]
    registry["experiments"][1]["scientific_status"] = "COMPLETE"
    registry["experiments"][1]["classification"] = "SEALED_RESULT"
    registry["experiments"][1]["required_artifacts"] = [{"ref": "feature", "path": "x.json", "sha256": "UNKNOWN"}]
    registry["experiments"][1]["result_status"] = "SEALED"
    graph["nodes"][0]["partial_evidence"] = ["partial.json"]
    graph["nodes"][1]["required_evidence"] = ["x.json"]
    errors = validate_graph(graph, registry)
    assert "complete child child_lane has incomplete parent parent_lane" in errors


def test_partial_result_cannot_pass_as_complete() -> None:
    base = {
        "experiment_id": "lane",
        "git_sha": "d" * 40,
        "branch": "main",
        "command": "python runner.py",
        "environment": {"os": "linux"},
        "package_versions": {"numpy": "2.2.6"},
        "hardware": {"cpu": "16 cores"},
        "seed": 42,
        "classification": "PARTIAL",
        "stdout_provenance": {"path": "stdout.log"},
        "stderr_provenance": {"path": "stderr.log"},
        "protocol_config_sha256": "3" * 64,
        "dataset_sha256": "4" * 64,
        "started_at_utc": "2026-08-25T00:00:00+00:00",
        "exit_status": "INTERRUPTED",
        "checkpoint_identity": {"path": "checkpoint.pt"},
        "partial_outputs": [{"path": "journal.jsonl"}],
        "failure_reason": "stopped after 164 surfaces",
    }
    assert validate_result_intake(base) == []
    partial_complete = copy.deepcopy(base)
    partial_complete["result_status"] = "COMPLETE"
    assert any("cannot claim COMPLETE" in error for error in validate_result_intake(partial_complete))


def test_complete_result_requires_outputs_hashes_and_metric_manifest() -> None:
    record = {
        "experiment_id": "lane",
        "git_sha": "d" * 40,
        "branch": "main",
        "command": "python runner.py",
        "environment": {},
        "package_versions": {},
        "hardware": {},
        "seed": 42,
        "classification": "COMPLETE",
        "stdout_provenance": {},
        "stderr_provenance": {},
    }
    errors = validate_result_intake(record)
    for field in ("output_files", "metric_manifest", "ended_at_utc"):
        assert any(field in error for error in errors)


def test_stale_document_detection_is_targeted(tmp_path: Path) -> None:
    registry = _minimal_registry()
    (tmp_path / "status.md").write_text("FINAL_10K_ABSENT\nparent absent\n", encoding="utf-8")
    findings = detect_stale_documents(registry, tmp_path)
    assert [item["rule"] for item in findings] == ["old_parent_claim"]


def test_generated_documents_are_deterministic_and_operational() -> None:
    registry = _minimal_registry()
    graph = _minimal_graph()
    first_matrix = build_execution_matrix(registry, graph)
    second_matrix = build_execution_matrix(registry, graph)
    first_handoff = build_tomorrow_handoff(registry)
    second_handoff = build_tomorrow_handoff(registry)
    assert first_matrix == second_matrix
    assert first_handoff == second_handoff
    for heading in ("CURRENT STATUS", "CAN RUN NOW?", "STOP GATE", "FORBIDDEN ACTIONS"):
        assert heading in first_matrix
    for heading in ("What is complete?", "What is currently partial?", "First actions tomorrow"):
        assert heading in first_handoff


def test_report_schema_orders_experiments_and_collects_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import research_control as rc

    registry = _minimal_registry()
    graph = _minimal_graph()
    (tmp_path / "status.md").write_text("parent absent\n", encoding="utf-8")
    monkeypatch.setattr(rc, "verify_git_identities", lambda *_: (["injected git failure"], {"branches": {}}))
    monkeypatch.setattr(rc, "verify_protected_scientific_state", lambda *_: ([], {"changed_scientific_paths": []}))
    report, errors = build_report(registry, graph, tmp_path)
    assert report["schema_version"] == "1.0"
    assert report["status"] == "FAIL"
    assert "injected git failure" in errors
    assert [item["id"] for item in report["experiments"]] == sorted(item["id"] for item in report["experiments"])
    assert report["errors"] == sorted(set(report["errors"]))
    assert json.dumps(report, sort_keys=True)


def test_preflight_blocks_wrong_branch_and_authorization_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import research_control as rc

    registry = _minimal_registry()
    graph = _minimal_graph()
    monkeypatch.setattr(
        rc,
        "verify_git_identities",
        lambda *_: ([], {"worktree": {"branch": "control", "head": "x" * 40}}),
    )
    errors, checks = preflight_lane("child_lane", registry, graph, tmp_path)
    assert checks["scientific_runner_invoked"] is False
    assert checks["authorization_gate"] == "needs review"
    assert any("preflight must run from feature" in error for error in errors)
    assert any("authorization gate" in error for error in errors)


def test_control_pair_detects_registry_graph_id_divergence() -> None:
    registry = _minimal_registry()
    registry["experiments"] = registry["experiments"][:1]
    graph = _minimal_graph()
    assert any("registry/graph id divergence" in error for error in validate_control_inputs(registry, graph))


def test_stale_scanner_reads_branch_only_document(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import research_control as rc

    registry = _minimal_registry()
    rule = {
        "id": "branch_only_stale",
        "file": "remote/status.md",
        "pattern": "old branch claim",
        "expected_current_experiment_commit": {"experiment": "child_lane", "commit": "c" * 40},
        "reason": "remote document is stale",
    }
    registry["stale_rules"] = [rule]
    monkeypatch.setattr(
        rc,
        "read_git_blob",
        lambda branch, path, root=b".": b"old branch claim\n" if branch == "feature" and path == rule["file"] else b"",
    )
    findings = detect_stale_documents(registry, tmp_path)
    assert [item["rule"] for item in findings] == ["branch_only_stale"]
