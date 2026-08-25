"""Machine-checkable research-state control utilities.

This module coordinates provenance only. It never executes a scientific
experiment, trains a model, prices a surface, or mutates evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


REGISTRY_PATH = Path("configs/research_experiment_registry.yaml")
GRAPH_PATH = Path("configs/research_dependency_graph.yaml")

CLASSIFICATIONS = {
    "SEALED_RESULT",
    "DEVELOPMENT_RESULT",
    "PROTOCOL_FROZEN",
    "IMPLEMENTATION_READY",
    "PARTIAL_CHECKPOINT",
    "PENDING",
    "BLOCKED",
    "SUPERSEDED_HISTORICAL",
}

SCIENTIFIC_STATUSES = {"COMPLETE", "PARTIAL", "PENDING", "NOT_STARTED", "UNKNOWN"}
INTAKE_CLASSIFICATIONS = {"COMPLETE", "PARTIAL", "FAILED"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

EXPERIMENT_FIELDS = (
    "id",
    "title",
    "classification",
    "scientific_status",
    "branch",
    "commit",
    "base_commit",
    "protocol",
    "protocol_sha256",
    "config",
    "config_sha256",
    "dataset",
    "dataset_sha256",
    "train_population",
    "validation_population",
    "test_population",
    "test_opened",
    "seeds",
    "runner",
    "compute_class",
    "execution_location",
    "intended_compute_environment",
    "execution_command",
    "dependencies",
    "required_artifacts",
    "partial_artifacts",
    "result_status",
    "test_exposure_state",
    "scientifically_legal_next_action",
    "forbidden_actions",
    "stop_gate",
)

INTAKE_BASE_FIELDS = (
    "experiment_id",
    "git_sha",
    "branch",
    "command",
    "environment",
    "package_versions",
    "hardware",
    "seed",
    "classification",
    "stdout_provenance",
    "stderr_provenance",
)

INTAKE_COMPLETE_FIELDS = (
    "protocol_config_sha256",
    "dataset_sha256",
    "started_at_utc",
    "ended_at_utc",
    "exit_status",
    "checkpoint_identity",
    "output_files",
    "metric_manifest",
)

INTAKE_PARTIAL_FIELDS = (
    "protocol_config_sha256",
    "dataset_sha256",
    "started_at_utc",
    "exit_status",
    "checkpoint_identity",
    "partial_outputs",
    "failure_reason",
)

SCIENTIFIC_OUTPUT_PREFIXES = (
    "data/",
    "evidence/",
    "outputs/",
    "market_data_audit/",
)


class ResearchControlError(ValueError):
    """Raised when the research control plane fails closed."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except OSError as exc:
        raise ResearchControlError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ResearchControlError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchControlError(f"{path} must contain a YAML object")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(GIT_SHA_RE.fullmatch(value))


def is_hash_placeholder(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    compact = value.replace(" ", "")
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_=;,./%+-]*", compact))


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _validate_artifact_list(
    owner: str,
    field: str,
    value: Any,
    errors: list[str],
) -> None:
    _require(isinstance(value, list), f"{owner}.{field} must be a list", errors)
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        label = f"{owner}.{field}[{index}]"
        _require(isinstance(item, dict), f"{label} must be an object", errors)
        if not isinstance(item, dict):
            continue
        for key in ("ref", "path", "sha256"):
            _require(key in item, f"{label} missing {key}", errors)
        if "path" in item:
            _require(
                isinstance(item.get("path"), str) and item["path"] and not Path(item["path"]).is_absolute(),
                f"{label}.path must be a nonempty repository-relative path",
                errors,
            )
        if "sha256" in item:
            _require(
                is_sha256(item.get("sha256")) or is_hash_placeholder(item.get("sha256")),
                f"{label}.sha256 must be SHA-256 or an explicit placeholder",
                errors,
            )


def validate_registry(registry: Any) -> list[str]:
    errors: list[str] = []
    _require(isinstance(registry, dict), "registry must be an object", errors)
    if not isinstance(registry, dict):
        return errors

    _require(registry.get("schema_version") == "1.0", "registry.schema_version must be '1.0'", errors)
    _require(
        registry.get("registry_kind") == "RESEARCH_EXPERIMENT_REGISTRY",
        "registry.registry_kind is invalid",
        errors,
    )
    snapshot = registry.get("snapshot")
    _require(isinstance(snapshot, dict), "registry.snapshot must be an object", errors)
    if isinstance(snapshot, dict):
        for key in ("observed_at_utc", "repository", "audit_base_commit", "control_branch", "remote_namespace"):
            _require(bool(snapshot.get(key)), f"registry.snapshot.{key} is required", errors)
        _require(is_git_sha(snapshot.get("audit_base_commit")), "audit_base_commit must be a Git SHA", errors)
        try:
            datetime.fromisoformat(str(snapshot.get("observed_at_utc")).replace("Z", "+00:00"))
        except ValueError:
            errors.append("registry.snapshot.observed_at_utc must be ISO-8601")

    experiments = registry.get("experiments")
    _require(isinstance(experiments, list) and bool(experiments), "registry.experiments must be a nonempty list", errors)
    if not isinstance(experiments, list):
        return errors

    by_id: dict[str, dict[str, Any]] = {}
    for index, experiment in enumerate(experiments):
        label = f"experiments[{index}]"
        _require(isinstance(experiment, dict), f"{label} must be an object", errors)
        if not isinstance(experiment, dict):
            continue
        missing = [field for field in EXPERIMENT_FIELDS if field not in experiment]
        _require(not missing, f"{label} missing fields: {', '.join(missing)}", errors)
        experiment_id = experiment.get("id")
        _require(
            isinstance(experiment_id, str) and bool(ID_RE.fullmatch(experiment_id)),
            f"{label}.id must be a lowercase snake_case identifier",
            errors,
        )
        if not isinstance(experiment_id, str):
            continue
        _require(experiment_id not in by_id, f"duplicate experiment id: {experiment_id}", errors)
        by_id[experiment_id] = experiment
        owner = f"experiment {experiment_id}"
        _require(experiment.get("classification") in CLASSIFICATIONS, f"{owner} has invalid classification", errors)
        _require(experiment.get("scientific_status") in SCIENTIFIC_STATUSES, f"{owner} has invalid scientific_status", errors)
        _require(isinstance(experiment.get("title"), str) and bool(experiment.get("title")), f"{owner}.title is required", errors)
        _require(isinstance(experiment.get("branch"), str) and bool(experiment.get("branch")), f"{owner}.branch is required", errors)
        _require(is_git_sha(experiment.get("base_commit")), f"{owner}.base_commit must be a Git SHA", errors)
        commit = experiment.get("commit")
        _require(is_git_sha(commit) or commit == "MISSING_REF", f"{owner}.commit must be a Git SHA or MISSING_REF", errors)
        _require(isinstance(experiment.get("test_opened"), bool), f"{owner}.test_opened must be boolean", errors)
        _require(isinstance(experiment.get("dependencies"), list), f"{owner}.dependencies must be a list", errors)
        _require(isinstance(experiment.get("forbidden_actions"), list) and bool(experiment.get("forbidden_actions")), f"{owner}.forbidden_actions must be nonempty", errors)
        for field in ("protocol_sha256", "config_sha256", "dataset_sha256"):
            _require(
                is_sha256(experiment.get(field)) or is_hash_placeholder(experiment.get(field)),
                f"{owner}.{field} must be SHA-256 or an explicit placeholder",
                errors,
            )
        _validate_artifact_list(experiment_id, "required_artifacts", experiment.get("required_artifacts"), errors)
        _validate_artifact_list(experiment_id, "partial_artifacts", experiment.get("partial_artifacts"), errors)

        status = experiment.get("scientific_status")
        classification = experiment.get("classification")
        if status == "COMPLETE":
            _require(classification in {"SEALED_RESULT", "DEVELOPMENT_RESULT", "SUPERSEDED_HISTORICAL"}, f"{owner}: COMPLETE requires a result or historical classification", errors)
            _require(bool(experiment.get("required_artifacts")), f"{owner}: COMPLETE requires evidence artifacts", errors)
            _require("PARTIAL" not in str(experiment.get("result_status")).upper(), f"{owner}: complete result_status mentions PARTIAL", errors)
        elif status == "PARTIAL":
            _require(classification == "PARTIAL_CHECKPOINT", f"{owner}: PARTIAL requires PARTIAL_CHECKPOINT", errors)
            _require(bool(experiment.get("partial_artifacts")), f"{owner}: PARTIAL requires partial artifacts", errors)
            _require("COMPLETE" not in str(experiment.get("result_status")).upper(), f"{owner}: partial result_status claims COMPLETE", errors)
        elif status in {"PENDING", "NOT_STARTED"}:
            _require(classification in {"PROTOCOL_FROZEN", "IMPLEMENTATION_READY", "PENDING", "BLOCKED"}, f"{owner}: pending/not-started classification is invalid", errors)
            if experiment.get("test_opened"):
                errors.append(f"{owner}: a pending/not-started experiment cannot have opened its test split")

    for experiment in by_id.values():
        for dependency in experiment.get("dependencies", []):
            _require(dependency in by_id, f"experiment {experiment['id']} references missing dependency {dependency}", errors)

    stale_rules = registry.get("stale_rules", [])
    _require(isinstance(stale_rules, list), "registry.stale_rules must be a list", errors)
    rule_ids: set[str] = set()
    for index, rule in enumerate(stale_rules if isinstance(stale_rules, list) else []):
        label = f"stale_rules[{index}]"
        _require(isinstance(rule, dict), f"{label} must be an object", errors)
        if not isinstance(rule, dict):
            continue
        rule_id = rule.get("id")
        _require(isinstance(rule_id, str) and rule_id not in rule_ids, f"{label}.id must be unique", errors)
        rule_ids.add(rule_id)
        for key in ("file", "pattern", "reason"):
            _require(isinstance(rule.get(key), str) and bool(rule.get(key)), f"{label}.{key} is required", errors)

    handoff = registry.get("tomorrow_handoff")
    _require(isinstance(handoff, dict), "registry.tomorrow_handoff must be an object", errors)
    if isinstance(handoff, dict):
        _require(isinstance(handoff.get("first_actions"), list) and bool(handoff.get("first_actions")), "tomorrow_handoff.first_actions must be nonempty", errors)
        _require(isinstance(handoff.get("human_review_gates"), list) and bool(handoff.get("human_review_gates")), "tomorrow_handoff.human_review_gates must be nonempty", errors)
    return errors


def validate_graph(graph: Any, registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(graph, dict), "dependency graph must be an object", errors)
    if not isinstance(graph, dict):
        return errors
    _require(graph.get("schema_version") == "1.0", "graph.schema_version must be '1.0'", errors)
    _require(graph.get("graph_kind") == "RESEARCH_DEPENDENCY_GRAPH", "graph.graph_kind is invalid", errors)
    nodes = graph.get("nodes")
    _require(isinstance(nodes, list) and bool(nodes), "graph.nodes must be a nonempty list", errors)
    if not isinstance(nodes, list):
        return errors

    experiments = {item.get("id"): item for item in registry.get("experiments", []) if isinstance(item, dict)}
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: dict[str, list[str]] = {}
    for index, node in enumerate(nodes):
        label = f"nodes[{index}]"
        _require(isinstance(node, dict), f"{label} must be an object", errors)
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        _require(isinstance(node_id, str) and bool(ID_RE.fullmatch(node_id)), f"{label}.id is invalid", errors)
        if not isinstance(node_id, str):
            continue
        _require(node_id not in nodes_by_id, f"duplicate graph node: {node_id}", errors)
        nodes_by_id[node_id] = node
        depends_on = node.get("depends_on")
        _require(isinstance(depends_on, list), f"node {node_id}.depends_on must be a list", errors)
        edges[node_id] = depends_on if isinstance(depends_on, list) else []
        _require(node.get("status_source") in experiments, f"node {node_id} has missing status_source", errors)
        _require(isinstance(node.get("runnable_now"), bool), f"node {node_id}.runnable_now must be boolean", errors)
        _require(isinstance(node.get("runnable_reason"), str) and bool(node.get("runnable_reason")), f"node {node_id}.runnable_reason is required", errors)
        _require(isinstance(node.get("required_evidence"), list), f"node {node_id}.required_evidence must be a list", errors)

    for node_id, dependencies in edges.items():
        for dependency in dependencies:
            _require(dependency in nodes_by_id, f"node {node_id} references missing dependency {dependency}", errors)

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node_id: str) -> None:
        mark = state.get(node_id, 0)
        if mark == 2:
            return
        if mark == 1:
            cycle_start = stack.index(node_id) if node_id in stack else 0
            errors.append("dependency cycle: " + " -> ".join([*stack[cycle_start:], node_id]))
            return
        state[node_id] = 1
        stack.append(node_id)
        for dependency in edges.get(node_id, []):
            if dependency in nodes_by_id:
                visit(dependency)
        stack.pop()
        state[node_id] = 2

    for node_id in sorted(edges):
        visit(node_id)

    for node_id, node in nodes_by_id.items():
        experiment = experiments.get(node.get("status_source"))
        if not experiment:
            continue
        child_status = experiment.get("scientific_status")
        if child_status == "COMPLETE":
            for dependency in node.get("depends_on", []):
                parent = experiments.get(dependency)
                if parent and parent.get("scientific_status") != "COMPLETE":
                    errors.append(f"complete child {node_id} has incomplete parent {dependency}")
        if experiment.get("classification") == "BLOCKED" and node.get("runnable_now"):
            errors.append(f"blocked lane {node_id} is marked runnable")
        declared = {item.get("path") for item in [*experiment.get("required_artifacts", []), *experiment.get("partial_artifacts", [])]}
        for evidence in node.get("required_evidence", []):
            if evidence not in declared:
                errors.append(f"node {node_id} evidence is absent from registry artifacts: {evidence}")
    return errors


def validate_result_intake(record: Any) -> list[str]:
    errors: list[str] = []
    _require(isinstance(record, dict), "result-intake record must be an object", errors)
    if not isinstance(record, dict):
        return errors
    missing_base = [field for field in INTAKE_BASE_FIELDS if field not in record]
    _require(not missing_base, f"missing base fields: {', '.join(missing_base)}", errors)
    classification = record.get("classification")
    _require(classification in INTAKE_CLASSIFICATIONS, "classification must be COMPLETE, PARTIAL, or FAILED", errors)
    _require(is_git_sha(record.get("git_sha")), "git_sha must be a 40-character Git SHA", errors)
    for field in ("environment", "package_versions", "hardware"):
        _require(isinstance(record.get(field), dict), f"{field} must be an object", errors)
    for field in ("stdout_provenance", "stderr_provenance"):
        _require(field in record, f"{field} is required", errors)
    if is_sha256(record.get("protocol_config_sha256")) is False and record.get("protocol_config_sha256") is not None:
        errors.append("protocol_config_sha256 must be SHA-256 when supplied")
    if is_sha256(record.get("dataset_sha256")) is False and record.get("dataset_sha256") is not None:
        errors.append("dataset_sha256 must be SHA-256 when supplied")

    if classification == "COMPLETE":
        missing = [field for field in INTAKE_COMPLETE_FIELDS if field not in record]
        _require(not missing, f"complete result missing fields: {', '.join(missing)}", errors)
        outputs = record.get("output_files")
        _require(isinstance(outputs, list) and bool(outputs), "complete result requires output_files", errors)
        if isinstance(outputs, list):
            for index, item in enumerate(outputs):
                _require(isinstance(item, dict) and bool(item.get("path")), f"output_files[{index}].path is required", errors)
                _require(is_sha256(item.get("sha256")) if isinstance(item, dict) else False, f"output_files[{index}].sha256 is invalid", errors)
    elif classification == "PARTIAL":
        missing = [field for field in INTAKE_PARTIAL_FIELDS if field not in record]
        _require(not missing, f"partial result missing fields: {', '.join(missing)}", errors)
        claims = str(record.get("scientific_claim_allowed", "")).upper()
        status = str(record.get("result_status", "")).upper()
        _require("COMPLETE" not in claims and "COMPLETE" not in status, "a partial result cannot claim COMPLETE", errors)
        _require(record.get("exit_status") != "SUCCESS", "a partial result cannot have SUCCESS exit status", errors)
    if classification == "FAILED":
        _require(bool(record.get("failure_reason")), "failed result requires failure_reason", errors)

    for timestamp_field in ("started_at_utc", "ended_at_utc"):
        if timestamp_field in record:
            try:
                datetime.fromisoformat(str(record[timestamp_field]).replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{timestamp_field} must be ISO-8601")
    return errors


def detect_stale_documents(registry: dict[str, Any], root: str | Path = ".") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    experiments = {item.get("id"): item for item in registry.get("experiments", []) if isinstance(item, dict)}
    base = Path(root)
    for rule in registry.get("stale_rules", []):
        path = base / rule["file"]
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        triggered = rule["pattern"] in text
        expected = rule.get("expected_current_experiment_commit")
        if isinstance(expected, dict):
            experiment = experiments.get(expected.get("experiment"))
            triggered = triggered and experiment is not None and experiment.get("commit") == expected.get("commit")
        else:
            complete_id = rule.get("when_complete_experiment")
            experiment = experiments.get(complete_id)
            triggered = triggered and experiment is not None and experiment.get("scientific_status") == "COMPLETE"
        if triggered:
            findings.append(
                {
                    "rule": rule["id"],
                    "file": rule["file"],
                    "pattern": rule["pattern"],
                    "classification": "STALE_AS_CURRENT_STATUS",
                    "reason": rule["reason"],
                }
            )
    return sorted(findings, key=lambda item: item["rule"])


def git_text(args: Iterable[str], root: str | Path = ".") -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ResearchControlError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout


def read_git_blob(ref: str, path: str, root: str | Path = ".") -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ResearchControlError(f"missing Git object {ref}:{path}")
    return completed.stdout


def verify_git_identities(registry: dict[str, Any], root: str | Path = ".") -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    details: dict[str, Any] = {"branches": {}, "identities": [], "worktree": {}}
    namespace = registry["snapshot"]["remote_namespace"]
    experiments = sorted(registry.get("experiments", []), key=lambda item: item["id"])
    for experiment in experiments:
        experiment_id = experiment["id"]
        branch = experiment["branch"]
        remote_ref = f"{namespace}/{branch}"
        branch_info: dict[str, Any] = {"remote_ref": remote_ref}
        try:
            remote_sha = git_text(["rev-parse", "--verify", f"refs/remotes/{remote_ref}^{{commit}}"], root).strip()
            branch_info["remote_sha"] = remote_sha
        except ResearchControlError:
            remote_sha = None
            branch_info["remote_sha"] = None
        details["branches"][experiment_id] = branch_info
        expected_commit = experiment["commit"]
        if expected_commit == "MISSING_REF":
            if remote_sha is not None:
                errors.append(f"{experiment_id}: registry expects missing ref but {remote_ref} exists at {remote_sha}")
        elif remote_sha is None:
            errors.append(f"{experiment_id}: missing remote ref {remote_ref}")
        elif remote_sha.lower() != expected_commit.lower():
            errors.append(f"{experiment_id}: {remote_ref} is {remote_sha}, registry expects {expected_commit}")

        if is_git_sha(expected_commit):
            try:
                git_text(["cat-file", "-e", f"{expected_commit}^{{commit}}"], root)
                details["branches"][experiment_id]["local_object_exists"] = True
            except ResearchControlError:
                details["branches"][experiment_id]["local_object_exists"] = False
                errors.append(f"{experiment_id}: commit object {expected_commit} is missing locally")
            try:
                merge_base = git_text(["merge-base", expected_commit, experiment["base_commit"]], root).strip()
                if merge_base.lower() != experiment["base_commit"].lower():
                    errors.append(f"{experiment_id}: base {experiment['base_commit']} is not an ancestor of {expected_commit}")
            except ResearchControlError:
                errors.append(f"{experiment_id}: ancestry check failed for {expected_commit}")

        identities = [
            ("config", experiment["branch"], experiment["config"], experiment.get("config_sha256")),
            ("dataset", experiment["branch"], experiment["dataset"], experiment.get("dataset_sha256")),
        ]
        for kind, ref, path, expected_hash in identities:
            if not is_sha256(expected_hash) or not path or "://" in path or path.startswith(("official ", "generated ", "future ", "committed ")):
                continue
            try:
                actual = sha256_bytes(read_git_blob(ref, path, root))
                details["identities"].append({"experiment": experiment_id, "kind": kind, "ref": ref, "path": path, "sha256": actual})
                if actual.lower() != expected_hash.lower():
                    errors.append(f"{experiment_id}: {kind} hash mismatch for {ref}:{path}")
            except ResearchControlError as exc:
                errors.append(f"{experiment_id}: {exc}")

        artifact_sources = [("required", x) for x in experiment.get("required_artifacts", [])]
        artifact_sources += [("partial", x) for x in experiment.get("partial_artifacts", [])]
        for role, artifact in artifact_sources:
            try:
                payload = read_git_blob(artifact["ref"], artifact["path"], root)
                actual = sha256_bytes(payload)
                expected = artifact.get("sha256")
                details["identities"].append({"experiment": experiment_id, "role": role, "ref": artifact["ref"], "path": artifact["path"], "sha256": actual})
                if is_sha256(expected) and actual.lower() != expected.lower():
                    errors.append(f"{experiment_id}: artifact hash mismatch {artifact['ref']}:{artifact['path']}")
            except ResearchControlError:
                errors.append(f"{experiment_id}: missing {role} artifact {artifact['ref']}:{artifact['path']}")

    try:
        status = git_text(["status", "--porcelain", "--untracked-files=no"], root)
        details["worktree"]["tracked_dirty_lines"] = len(status.splitlines())
        if registry["snapshot"].get("tracked_tree_required_clean") and status.strip():
            errors.append("tracked working tree is not clean")
    except ResearchControlError as exc:
        errors.append(str(exc))
    try:
        current_branch = git_text(["branch", "--show-current"], root).strip()
        head = git_text(["rev-parse", "HEAD"], root).strip()
        details["worktree"].update({"branch": current_branch, "head": head})
        if current_branch != registry["snapshot"]["control_branch"]:
            errors.append(f"current branch {current_branch} does not match control branch {registry['snapshot']['control_branch']}")
        base = registry["snapshot"]["audit_base_commit"]
        merge_base = git_text(["merge-base", head, base], root).strip()
        if merge_base.lower() != base.lower():
            errors.append(f"control HEAD is not based on audit base {base}")
    except ResearchControlError as exc:
        errors.append(str(exc))
    return errors, details


def verify_protected_scientific_state(root: str | Path = ".") -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    changed: list[str] = []
    for prefix in SCIENTIFIC_OUTPUT_PREFIXES:
        try:
            lines = git_text(["diff", "--name-only", "HEAD", "--", prefix], root).splitlines()
        except ResearchControlError as exc:
            errors.append(str(exc))
            continue
        changed.extend(lines)
    if changed:
        errors.append("protected scientific output/evidence paths have working-tree changes")
    return errors, {"changed_scientific_paths": sorted(changed)}


def build_execution_matrix(registry: dict[str, Any], graph: dict[str, Any]) -> str:
    experiments = {item["id"]: item for item in registry.get("experiments", []) if isinstance(item, dict)}
    nodes = {item["id"]: item for item in graph.get("nodes", []) if isinstance(item, dict)}
    lines = [
        "# Research Execution Matrix",
        "",
        "Generated deterministically from the registry and dependency graph.",
        "",
        f"- Snapshot: `{registry['snapshot']['observed_at_utc']}`",
        f"- Audit base: `{registry['snapshot']['audit_base_commit']}`",
        "- CAN RUN NOW applies to this coordination run; a fresh human authorization is separately required where stated.",
        "",
    ]
    for experiment_id in sorted(set(experiments) & set(nodes)):
        experiment = experiments[experiment_id]
        node = nodes[experiment_id]
        lines.extend(
            [
                f"## {experiment_id}",
                "",
                f"- CURRENT STATUS: {experiment['classification']} / {experiment['scientific_status']} / {experiment['result_status']}",
                f"- TEST EXPOSURE: {experiment['test_exposure_state']} (`test_opened={str(experiment['test_opened']).lower()}`)",
                f"- CAN RUN NOW? {'YES' if node['runnable_now'] else 'NO'}",
                f"- WHY / WHY NOT? {node['runnable_reason']}",
                f"- DEPENDENCIES: {', '.join(node['depends_on']) if node['depends_on'] else 'none'}",
                f"- WORKTREE / BRANCH: `{experiment['branch']}` at `{experiment['commit']}`; base `{experiment['base_commit']}`",
                f"- RUNNER: `{experiment['runner']}`",
                f"- COMPUTE CLASS: {experiment['compute_class']}",
                f"- RECOMMENDED LOCATION: {experiment['execution_location']}",
                f"- EXPECTED SCALE: train={experiment['train_population']}; validation={experiment['validation_population']}; test={experiment['test_population']}; compute={experiment['compute_class']}",
                f"- REQUIRED INPUT IDENTITIES: config=`{experiment['config']}` ({experiment['config_sha256']}); dataset=`{experiment['dataset']}` ({experiment['dataset_sha256']})",
                f"- STOP GATE: {experiment['stop_gate']}",
                "- FORBIDDEN ACTIONS:",
            ]
        )
        lines.extend(f"  - {action}" for action in experiment["forbidden_actions"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_tomorrow_handoff(registry: dict[str, Any]) -> str:
    experiments = sorted(registry.get("experiments", []), key=lambda item: item["id"])
    handoff = registry["tomorrow_handoff"]
    complete = [item for item in experiments if item["scientific_status"] == "COMPLETE"]
    partial = [item for item in experiments if item["scientific_status"] == "PARTIAL"]
    remaining = [item for item in experiments if item["scientific_status"] != "COMPLETE"]
    gpu = [item["id"] for item in remaining if "GPU" in item["compute_class"].upper()]
    multicore = list(dict.fromkeys(item["id"] for item in remaining if "MULTICORE" in item["compute_class"].upper()))
    light = [item["id"] for item in remaining if "LIGHT_LOCAL" in item["compute_class"].upper()]
    blocked = [item["id"] for item in experiments if item["classification"] == "BLOCKED" or item["scientific_status"] == "NOT_STARTED"]
    lines = [
        "# Next Research Execution Handoff",
        "",
        f"Factual snapshot generated from the registry at `{handoff['remote_snapshot_utc']}`.",
        "",
        "## What is complete?",
        "",
    ]
    lines.extend(f"- `{item['id']}` — {item['result_status']}" for item in complete)
    lines.extend(
        [
            "",
            "## What changed on remote branches?",
            "",
            f"The live audit found changes beyond `{registry['snapshot']['audit_base_commit']}`. Rerun `git fetch origin` and this verifier before acting.",
            "",
            "| Lane | Remote branch | Observed tip | Status |",
            "|---|---|---|---|",
        ]
    )
    changed = [
        item
        for item in experiments
        if item["branch"] != registry["snapshot"]["control_branch"]
        and not item["branch"].startswith("MISSING_REMOTE_BRANCH")
        and item["commit"] != "MISSING_REF"
        and item["commit"] != registry["snapshot"]["audit_base_commit"]
    ]
    for item in sorted(changed, key=lambda entry: entry["id"]):
        lines.append(
            f"| `{item['id']}` | `{item['branch']}` | `{item['commit']}` | {item['scientific_status']} / {item['result_status']} |"
        )
    lines.extend(
        [
            "",
            "## What is currently partial?",
            "",
        ]
    )
    lines.extend(f"- `{item['id']}` — {item['result_status']}" for item in partial)
    lines.extend(
        [
            "",
            "## What can run immediately tomorrow?",
            "",
            "Nothing scientific may start without the explicit review gate below. Coordination/read-only checks can run immediately.",
            "",
            "## Compute classes",
            "",
            f"- REQUIRES GPU: {', '.join(f'`{item}`' for item in gpu) if gpu else 'None.'}",
            f"- REQUIRES MULTICORE CPU: {', '.join(f'`{item}`' for item in multicore) if multicore else 'None.'}",
            f"- LIGHT LOCAL AFTER AUTHORIZATION: {', '.join(f'`{item}`' for item in light) if light else 'None.'}",
            f"- REMAINS BLOCKED: {', '.join(f'`{item}`' for item in blocked) if blocked else 'None.'}",
            "",
            "## Review gates before each experiment",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in handoff["human_review_gates"])
    lines.extend(
        [
            "",
            "## First actions tomorrow",
            "",
        ]
    )
    lines.extend(f"{index}. {item}" for index, item in enumerate(handoff["first_actions"], 1))
    return "\n".join(lines).rstrip() + "\n"


def build_report(registry: dict[str, Any], graph: dict[str, Any], root: str | Path = ".") -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    registry_errors = validate_registry(registry)
    graph_errors = validate_graph(graph, registry)
    errors.extend(registry_errors)
    errors.extend(graph_errors)
    git_errors, git_details = verify_git_identities(registry, root)
    protected_errors, protected_details = verify_protected_scientific_state(root)
    errors.extend(git_errors)
    errors.extend(protected_errors)
    stale = detect_stale_documents(registry, root)
    experiments = sorted(registry.get("experiments", []), key=lambda item: item["id"])
    report = {
        "schema_version": "1.0",
        "report_kind": "RESEARCH_STATE_VERIFICATION_REPORT",
        "status": "PASS" if not errors else "FAIL",
        "as_of": registry["snapshot"]["observed_at_utc"],
        "summary": {
            "experiment_count": len(experiments),
            "complete": sum(item["scientific_status"] == "COMPLETE" for item in experiments),
            "partial": sum(item["scientific_status"] == "PARTIAL" for item in experiments),
            "pending_or_not_started": sum(item["scientific_status"] in {"PENDING", "NOT_STARTED"} for item in experiments),
            "error_count": len(errors),
            "stale_document_count": len(stale),
        },
        "errors": sorted(errors),
        "experiments": [
            {
                "id": item["id"],
                "classification": item["classification"],
                "scientific_status": item["scientific_status"],
                "branch": item["branch"],
                "commit": item["commit"],
                "result_status": item["result_status"],
                "test_exposure_state": item["test_exposure_state"],
            }
            for item in experiments
        ],
        "dependency_graph": {
            "node_count": len(graph.get("nodes", [])),
            "edge_count": sum(len(item.get("depends_on", [])) for item in graph.get("nodes", []) if isinstance(item, dict)),
        },
        "git": git_details,
        "protected_scientific_state": protected_details,
        "stale_documents": stale,
    }
    return report, sorted(set(errors))


def write_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_control_inputs(root: str | Path = ".") -> tuple[dict[str, Any], dict[str, Any]]:
    base = Path(root)
    return load_yaml(base / REGISTRY_PATH), load_yaml(base / GRAPH_PATH)


def preflight_lane(
    lane_id: str,
    registry: dict[str, Any],
    graph: dict[str, Any],
    root: str | Path = ".",
) -> tuple[list[str], dict[str, Any]]:
    """Check identities and host readiness without invoking any runner."""
    errors: list[str] = []
    experiment_ids = {item["id"] for item in registry.get("experiments", []) if isinstance(item, dict)}
    if lane_id not in experiment_ids:
        raise ResearchControlError(f"unknown experiment id {lane_id}")
    errors.extend(validate_registry(registry))
    errors.extend(validate_graph(graph, registry))
    experiment = next(item for item in registry["experiments"] if item["id"] == lane_id)
    node = next((item for item in graph.get("nodes", []) if item.get("id") == lane_id), None)

    checks: dict[str, Any] = {
        "branch": experiment["branch"],
        "expected_commit": experiment["commit"],
        "compute_class": experiment["compute_class"],
        "runner": experiment["runner"],
        "scientific_runner_invoked": False,
    }

    git_errors, git_details = verify_git_identities(registry, root)
    errors.extend(git_errors)
    checks["worktree"] = git_details.get("worktree", {})
    current_branch = checks["worktree"].get("branch")
    if current_branch != experiment["branch"]:
        errors.append(
            f"preflight must run from {experiment['branch']}, not {current_branch}; refusing to improvise a checkout"
        )

    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import yaml  # noqa: F401

        checks["required_python_modules"] = "PASS"
    except Exception as exc:  # host-readiness only; never substitute a scientific dependency
        checks["required_python_modules"] = f"FAIL: {type(exc).__name__}"
        errors.append(f"required Python modules unavailable: {exc}")

    compute_upper = experiment["compute_class"].upper()
    if "GPU" in compute_upper:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            checks["cuda_available"] = cuda_available
            if not cuda_available:
                errors.append("GPU lane requires CUDA availability")
        except Exception as exc:
            checks["cuda_available"] = f"FAIL: {type(exc).__name__}"
            errors.append(f"PyTorch/CUDA readiness unavailable: {exc}")
    else:
        checks["cuda_required"] = False

    if "MULTICORE" in compute_upper or "16" in str(experiment.get("intended_compute_environment", "")):
        cores = os.cpu_count() or 0
        checks["logical_cpu_count"] = cores
        if cores < 2:
            errors.append("CPU lane requires at least two logical CPUs")

    if node is None:
        errors.append(f"dependency-graph node is missing: {lane_id}")
    elif node.get("runnable_now") is False:
        checks["authorization_gate"] = node.get("runnable_reason")
        errors.append(f"preflight stops at authorization gate: {node['runnable_reason']}")

    return sorted(set(errors)), checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the Double Heston research control plane without running science.")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--json", metavar="PATH", help="Write the structured verification report to PATH")
    parser.add_argument("--quiet", action="store_true", help="Print only the final status and error count")
    parser.add_argument("--preflight", metavar="EXPERIMENT_ID", help="Build a no-science launch-readiness checklist for a lane")
    parser.add_argument("--scan-stale", action="store_true", help="Scan configured documents for obvious stale current-status language")
    parser.add_argument("--generate-matrix", metavar="PATH", help="Generate the execution matrix Markdown")
    parser.add_argument("--generate-handoff", metavar="PATH", help="Generate tomorrow's handoff Markdown")
    parser.add_argument("--validate-result-intake", metavar="PATH", help="Validate a JSON/YAML result-intake record")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        registry, graph = load_control_inputs(root)
        if args.validate_result_intake:
            record = load_yaml(args.validate_result_intake) if str(args.validate_result_intake).endswith((".yaml", ".yml")) else json.loads(Path(args.validate_result_intake).read_text(encoding="utf-8"))
            errors = validate_result_intake(record)
            print("RESULT_INTAKE_PASS" if not errors else "RESULT_INTAKE_FAIL")
            for error in errors:
                print(f"ERROR: {error}")
            return 0 if not errors else 1
        if args.scan_stale:
            findings = detect_stale_documents(registry, root)
            print(json.dumps(findings, indent=2, sort_keys=True))
            return 0
        if args.generate_matrix:
            Path(args.generate_matrix).write_text(build_execution_matrix(registry, graph), encoding="utf-8")
            print(f"wrote {args.generate_matrix}")
        if args.generate_handoff:
            Path(args.generate_handoff).write_text(build_tomorrow_handoff(registry), encoding="utf-8")
            print(f"wrote {args.generate_handoff}")
        if args.preflight:
            errors, checks = preflight_lane(args.preflight, registry, graph, root)
            print(
                json.dumps(
                    {"experiment": args.preflight, "pass": not errors, "errors": errors, "checks": checks},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if not errors else 1

        report, errors = build_report(registry, graph, root)
        if args.json:
            write_report(report, args.json)
        if not args.quiet:
            print(report["status"])
            print(f"experiments={report['summary']['experiment_count']} complete={report['summary']['complete']} partial={report['summary']['partial']} stale={report['summary']['stale_document_count']}")
            for error in errors:
                print(f"ERROR: {error}")
        else:
            print(f"{report['status']} errors={len(errors)}")
        return 0 if not errors else 1
    except (ResearchControlError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
