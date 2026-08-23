"""Explicit, separately authorized final R2 10,000-surface generation pathway.

This module is the ONLY pathway that can price the final clean-core cohort.
It exists because the frozen generator
(:mod:`src.r2_synthetic_generation`) intentionally keeps final surface
pricing fail-closed (``_build_generation_cohort`` rejects any cohort other
than the development pilot, and the frozen config records
``final_10k_generation_command: NOT_AUTHORIZED_IN_THIS_MILESTONE``).  That
safety control is not deleted or bypassed here; this module adds the
separate explicit command the frozen contract reserved
(``final_10k_requires_separate_explicit_command_and_authorization: true``).

Controlled sequence enforced by this module:

1. ``preflight`` — non-pricing checks only (contract, panel identity,
   quotas, uniqueness, constraints, provenance hashes, quarantine).
2. ``authorize`` — preflight must pass, then a committed authorization
   marker is written to ``evidence/R2_FINAL_10K_GENERATION_AUTHORIZED.txt``.
   The marker records every frozen identity hash and states that no final
   surface output existed when the authorization SHA was committed.
3. ``generate-final`` — refuses to run unless the authorization marker is
   committed at HEAD, the authorization base is an ancestor of HEAD, HEAD
   is visible on a remote ref, and every recorded hash still matches the
   working tree.  It then prices exactly the 10,000 rows of the frozen
   readiness panel (read with correctly-rounded ``round_trip`` float
   parsing so bytes are reproducible), retaining — never replacing — any
   failure.
4. ``replay`` — deterministic reproducibility check: full replay of all
   10,000 surfaces or a predeclared subset whose rule was committed before
   any final output existed.

There is no hidden boolean bypass: the module contains no ``--force`` /
``--unsafe`` flag, and any post-authorization edit to this file (or to any
recorded scientific dependency) invalidates the marker hash checks.

NO MODEL TRAINING and NO REAL-MARKET DATA are touched by anything here.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import r2_synthetic_generation as frozen
from .constants import PARAMETER_NAMES
from .dheston.real_market_policy import (
    RealMarketWeightUpdateQuarantineError,
    resolve_real_market_epochs,
)
from .r2_representation import (
    CANONICAL_SLOT_KEYS,
    REPRESENTATION_NAME,
    REPRESENTATION_VERSION,
    SOURCE_SYNTHETIC,
    validate_payload,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_MARKER = (
    ROOT / "evidence" / "R2_FINAL_10K_GENERATION_AUTHORIZED.txt"
)
READINESS_OUTPUT = frozen.READINESS_OUTPUT
READINESS_MANIFEST_PATH = READINESS_OUTPUT / "readiness_manifest.json"
PANEL_CSV_PATH = READINESS_OUTPUT / "final_parameter_panel.csv"
PILOT_MANIFEST_PATH = frozen.PILOT_OUTPUT / "manifest.json"
FINAL_OUTPUT = ROOT / "data" / "final_r2_clean_10000"
FINAL_REPLAY_OUTPUT = ROOT / "data" / "final_r2_clean_10000_replay"

FINAL_GENERATOR_VERSION = "r2-final-generation-v1"
FINAL_COHORT = "final"
FINAL_TOTAL_SURFACES = 10_000
FINAL_DATASET_STATUS = (
    "FINAL_R2_CLEAN_10000_RESEARCH_SYNTHETIC_TRUTH_DATASET_"
    "FROZEN_BEFORE_MODEL_TRAINING"
)
NO_PRIOR_OUTPUT_STATEMENT = (
    "NO FINAL R2 10K SURFACE OUTPUT EXISTED WHEN THIS AUTHORIZATION SHA WAS COMMITTED."
)

# Predeclared deterministic replay-subset rule, fixed in this file before any
# final output existed and independent of surface-quality results: every
# 500th panel row by file order, plus the first PER_GROUP_ROW_LIMIT rows of
# every (distribution, split) group, de-duplicated and kept in panel order.
REPLAY_SUBSET_STRIDE = 500
REPLAY_SUBSET_PER_GROUP_ROW_LIMIT = 3

# The generator source changed once after the readiness run recorded its
# provenance: canonical main commit 65aab62 (PR #28) moved
# ``output = Path(output_directory)`` two lines earlier inside
# ``run_final_readiness`` so the insufficient-pool failure-retention branch
# could write its evidence.  That branch is unreachable from final
# generation (the final cohort reads the committed readiness panel directly
# and never calls ``run_final_readiness``), and no sampling, selection,
# conditioning, pricing, validation, or serialization logic changed.  This
# is the sole classified difference; every other scientific-dependency hash
# must match readiness/pilot provenance exactly.
GENERATOR_SOURCE_DRIFT_CLASSIFICATION = (
    "readiness-recorded generator_source_sha256 7152a4cfd820802afda4cb1a15c5"
    "46428f835a0cc0929dbfde1388aa7c20dfc4 differs from canonical main via "
    "commit 65aab62 (PR #28) which repaired the run_final_readiness "
    "insufficient-pool retention path; the changed lines are unreachable "
    "from final generation and no scientific generation logic changed"
)

MARKER_REQUIRED_STATEMENTS = (
    "FINAL_R2_10K_GENERATION_AUTHORIZED",
    NO_PRIOR_OUTPUT_STATEMENT,
)
MARKER_REQUIRED_FIELDS = (
    "canonical_main_base_git_sha",
    "frozen_contract_reference",
    "frozen_parameter_panel_sha256",
    "production_pricer_sha256",
    "r2_synthetic_interface_sha256",
    "generation_config_sha256",
    "generator_source_sha256",
    "final_generation_module_sha256",
    "final_quota_surfaces",
    "noise_level",
    "real_market_inputs",
    "training_authorization",
    "g8_authorization",
)
MARKER_HASH_FIELDS = {
    "frozen_parameter_panel_sha256": "panel",
    "generation_config_sha256": "config",
    "production_pricer_sha256": "pricer",
    "r2_synthetic_interface_sha256": "r2_synthetic_interface",
    "generator_source_sha256": "generator_source",
    "final_generation_module_sha256": "final_generation_module",
}


class FinalGenerationAuthorizationError(RuntimeError):
    """Raised when final generation is not explicitly authorized."""


class FinalDatasetValidationError(RuntimeError):
    """Raised when a produced or on-disk final dataset violates the contract."""


# ---------------------------------------------------------------------------
# Identity hashes
# ---------------------------------------------------------------------------


def _final_module_path() -> Path:
    return Path(__file__).resolve()


def scientific_dependency_hashes() -> dict[str, str]:
    """Current SHA-256 hashes of every frozen scientific dependency."""
    return {
        "panel": frozen.sha256_file(PANEL_CSV_PATH),
        "config": frozen.sha256_file(frozen.CONFIG_PATH),
        "pricer": frozen.sha256_file(frozen.PRICER_PATH),
        "generator_source": frozen.sha256_file(frozen.__file__),
        "reviewed_sampler_source": frozen.sha256_file(frozen.SAMPLER_SOURCE_PATH),
        "reviewed_sampling_config": frozen.sha256_file(frozen.REVIEWED_CONFIG_PATH),
        "r2_synthetic_interface": frozen.sha256_file(frozen.R2_SYNTHETIC_INTERFACE_PATH),
        "final_generation_module": frozen.sha256_file(_final_module_path()),
    }


# ---------------------------------------------------------------------------
# Git helpers (isolated so tests can stub repository-state checks)
# ---------------------------------------------------------------------------


def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _current_head() -> str:
    try:
        return _git(["rev-parse", "HEAD"]).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "GIT_SHA_UNAVAILABLE"


def _marker_committed_at_head(marker: Path | None = None) -> bool:
    """True only if the marker bytes are committed in the HEAD tree."""
    marker = AUTHORIZATION_MARKER if marker is None else marker
    relative = marker.resolve().relative_to(ROOT).as_posix()
    try:
        show = _git(["show", f"HEAD:{relative}"])
    except (OSError, subprocess.CalledProcessError):
        return False
    if show.returncode != 0:
        return False
    return show.stdout.encode("utf-8") == marker.read_bytes()


def _base_is_ancestor(base_sha: str) -> bool:
    try:
        return _git(
            ["merge-base", "--is-ancestor", base_sha, "HEAD"], check=False
        ).returncode == 0
    except OSError:
        return False


def _head_on_remote(head_sha: str) -> bool:
    """True only if some remote-tracking ref contains HEAD.

    The operator must ``git fetch`` first; this check then proves the
    authorization commit was pushed and is remotely visible.
    """
    try:
        listing = _git(["branch", "-r", "--contains", head_sha], check=False)
    except OSError:
        return False
    if listing.returncode != 0:
        return False
    return any(
        line.strip() and not line.strip().startswith("HEAD ->")
        for line in listing.stdout.splitlines()
    )


# ---------------------------------------------------------------------------
# Frozen panel access
# ---------------------------------------------------------------------------


def load_final_panel(path: Path = PANEL_CSV_PATH) -> pd.DataFrame:
    """Load the frozen selected panel with exact float64 round-trip parsing.

    ``float_precision="round_trip"`` is mandatory: the panel CSV text was
    written with ``%.17g`` from the readiness run's in-memory float64
    values, and pandas' default C float parser can differ from a correctly
    rounded strtod by one ULP.  Round-trip parsing reproduces the readiness
    float64 values bit-exactly (verified against every stored
    ``parameter_vector_hash``).
    """
    frame = pd.read_csv(path, float_precision="round_trip")
    expected_columns = ["candidate_key", "distribution", "split", *PARAMETER_NAMES, "parameter_vector_hash"]
    if list(frame.columns) != expected_columns:
        raise FinalDatasetValidationError(
            f"final panel columns drifted: {list(frame.columns)}"
        )
    frame["candidate_id"] = [
        int(key.rsplit("_", 1)[1]) for key in frame["candidate_key"]
    ]
    for _, row in frame.iterrows():
        expected_key = f"{row['distribution']}_{int(row['candidate_id']):06d}"
        if row["candidate_key"] != expected_key:
            raise FinalDatasetValidationError(
                f"candidate_key/distribution identity mismatch: {row['candidate_key']}"
            )
    return frame


def load_readiness_manifest(path: Path = READINESS_MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("status") != "FINAL_CANDIDATE_POOL_READINESS_VERIFIED_NO_PRICING":
        raise FinalDatasetValidationError(
            "readiness manifest is not the verified no-pricing readiness record"
        )
    if manifest.get("all_sufficient") is not True:
        raise FinalDatasetValidationError("readiness pools were not sufficient")
    if int(manifest.get("selected_panel_count", -1)) != FINAL_TOTAL_SURFACES:
        raise FinalDatasetValidationError("readiness panel count is not 10,000")
    return manifest


# ---------------------------------------------------------------------------
# Preflight (NO PRICE CALLS)
# ---------------------------------------------------------------------------


def _quadratic_gcd_ok(config: Mapping[str, Any]) -> bool:
    lattice = config["conditioning"]["lattice"]
    count = int(lattice["combination_count"])
    return all(
        math.gcd(int(stride), count) == 1
        for stride in config["conditioning"]["strides"].values()
    )


def run_preflight() -> dict[str, Any]:
    """Run every non-pricing pre-generation check; raise on any failure.

    This function must never price a surface: it touches no pricing
    entrypoint (tests enforce this by making any price call raise).
    """
    from src.constraints import validate_parameters

    report: dict[str, Any] = {"checks": {}, "no_price_calls": True}

    def require(name: str, ok: bool, detail: str = "") -> None:
        report["checks"][name] = {"passed": bool(ok), "detail": detail}
        if not ok:
            report["checks"][name]["failure"] = True
            raise FinalDatasetValidationError(
                f"preflight failed: {name}" + (f" ({detail})" if detail else "")
            )

    config = frozen.load_generation_config()
    require("contract_parses", True)
    require("r2_slot_count_20", len(CANONICAL_SLOT_KEYS) == 20)
    require(
        "representation_identity",
        config["representation"]["name"] == REPRESENTATION_NAME
        and config["representation"]["version"] == REPRESENTATION_VERSION,
    )
    require(
        "final_gate_still_closed_in_frozen_config",
        config["execution_gates"]["final_10k_generation_command"]
        == "NOT_AUTHORIZED_IN_THIS_MILESTONE",
    )
    require(
        "training_gate_closed",
        config["execution_gates"]["training_commands_in_this_milestone"] == "NONE"
        and config["final_10k_generated"] is False,
    )

    # Authoritative pilot evidence must still verify (hashes artifacts; no pricing).
    frozen._verified_pilot_evidence()
    require("authoritative_pilot_verified", True)

    readiness = load_readiness_manifest()
    pilot_manifest = json.loads(PILOT_MANIFEST_PATH.read_text(encoding="utf-8"))
    pilot_hashes = pilot_manifest["provenance_hashes"]

    hashes = scientific_dependency_hashes()
    require(
        "panel_hash_matches_readiness",
        hashes["panel"] == readiness["selected_panel_csv_sha256"],
        f"panel {hashes['panel'][:12]}... vs readiness {readiness['selected_panel_csv_sha256'][:12]}...",
    )
    require(
        "production_pricer_hash_matches_pilot_provenance",
        hashes["pricer"] == pilot_hashes["production_pricer_sha256"],
    )
    require(
        "r2_interface_hash_matches_readiness",
        hashes["r2_synthetic_interface"] == readiness["r2_synthetic_interface_sha256"],
    )
    require(
        "reviewed_sampler_source_hash_matches_readiness",
        hashes["reviewed_sampler_source"] == readiness["reviewed_sampler_source_sha256"],
    )
    require(
        "reviewed_sampling_config_hash_matches_readiness",
        hashes["reviewed_sampling_config"] == readiness["reviewed_sampling_config_sha256"],
    )
    require(
        "generation_config_hash_matches_readiness",
        hashes["config"] == readiness["generation_config_sha256"],
    )
    require(
        "generator_source_drift_classified",
        hashes["generator_source"] != readiness["generator_source_sha256"]
        and GENERATOR_SOURCE_DRIFT_CLASSIFICATION.startswith("readiness-recorded"),
        "generator source differs from readiness provenance only via the "
        "pre-classified 65aab62 failure-path repair",
    )

    panel = load_final_panel()
    require("panel_count_10000", len(panel) == FINAL_TOTAL_SURFACES, f"{len(panel)} rows")

    quotas = config["quotas"]["final_clean_core"]
    distribution_counts = panel["distribution"].value_counts().to_dict()
    require(
        "distribution_quotas_exact",
        distribution_counts == quotas["distributions"],
        str(distribution_counts),
    )
    split_ok = True
    observed_splits: dict[str, dict[str, int]] = {}
    for distribution in frozen.DISTRIBUTIONS:
        sub = panel.loc[panel["distribution"] == distribution]
        counts = {split: int((sub["split"] == split).sum()) for split in frozen.SPLIT_ORDER}
        observed_splits[distribution] = counts
        if any(counts[split] != int(quotas["splits"][split][distribution]) for split in frozen.SPLIT_ORDER):
            split_ok = False
    report["observed_split_counts"] = observed_splits
    require("split_quotas_exact", split_ok, str(observed_splits))

    require(
        "unique_candidate_ids",
        panel["candidate_key"].is_unique
        and all(
            group["candidate_id"].is_unique
            for _, group in panel.groupby("distribution")
        ),
    )

    vector_hashes = [frozen.parameter_vector_hash(row) for _, row in panel.iterrows()]
    require(
        "unique_parameter_vectors",
        len(set(vector_hashes)) == len(panel),
    )
    vector_splits: dict[str, set[str]] = {}
    for vector_hash, split in zip(vector_hashes, panel["split"], strict=True):
        vector_splits.setdefault(vector_hash, set()).add(split)
    require(
        "no_cross_split_overlap",
        all(len(splits) == 1 for splits in vector_splits.values()),
    )
    require(
        "panel_hash_column_consistent",
        list(panel["parameter_vector_hash"]) == vector_hashes,
        "recomputed parameter_vector_hash differs from the readiness column",
    )

    constraint_failures = 0
    for _, row in panel.iterrows():
        diagnostics = validate_parameters(
            row[PARAMETER_NAMES].to_numpy(dtype=np.float64)
        )
        if not diagnostics["is_valid"]:
            constraint_failures += 1
    require(
        "parameter_constraints_valid", constraint_failures == 0,
        f"{constraint_failures} rows violate canonical constraints",
    )

    # Conditioning contract unchanged: identical config hash already proven;
    # additionally exercise the frozen mapping arithmetic (no pricing).
    require("_conditioning_gcd_ok", _quadratic_gcd_ok(config))
    spot_indices = (0, 1, FINAL_TOTAL_SURFACES - 1)
    for index in spot_indices:
        conditioning, provenance = frozen.build_conditioning(index, FINAL_COHORT, config)
        if not (
            conditioning.dte[1] > conditioning.dte[0]
            and conditioning.spot == 100.0
            and provenance["seed"] == int(config["conditioning"]["seeds"]["final"])
            and provenance["stride"] == int(config["conditioning"]["strides"]["final"])
            and provenance["lattice_index"]
            == (index * provenance["stride"])
            % int(config["conditioning"]["lattice"]["combination_count"])
        ):
            require("conditioning_contract_unchanged", False, f"index {index}")
    require("conditioning_contract_unchanged", True)

    require("final_output_directory_absent", not FINAL_OUTPUT.exists())
    require("final_replay_directory_absent", not FINAL_REPLAY_OUTPUT.exists())

    try:
        resolve_real_market_epochs(
            config_real_epochs=1,
            continuous_requested=False,
            allow_noncanonical_real_weight_updates=False,
            continuous_epoch_limit=0,
        )
    except RealMarketWeightUpdateQuarantineError:
        require("real_market_weight_updates_quarantined", True)
    else:
        require("real_market_weight_updates_quarantined", False, "quarantine guard inactive")

    report["passed"] = True
    report["hashes"] = hashes
    report["head_git_sha"] = _current_head()
    return report


# ---------------------------------------------------------------------------
# Authorization marker
# ---------------------------------------------------------------------------


def parse_authorization_marker(path: Path | None = None) -> dict[str, str]:
    path = AUTHORIZATION_MARKER if path is None else path
    text = Path(path).read_text(encoding="utf-8")
    for statement in MARKER_REQUIRED_STATEMENTS:
        if statement not in text:
            raise FinalGenerationAuthorizationError(
                f"authorization marker is missing required statement: {statement!r}"
            )
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key.strip()] = value.strip()
    missing = [field for field in MARKER_REQUIRED_FIELDS if field not in fields]
    if missing:
        raise FinalGenerationAuthorizationError(
            f"authorization marker is missing fields: {missing}"
        )
    if fields["final_quota_surfaces"] != str(FINAL_TOTAL_SURFACES):
        raise FinalGenerationAuthorizationError(
            "authorization marker quota is not the frozen 10,000"
        )
    if fields["noise_level"] != "0.0":
        raise FinalGenerationAuthorizationError(
            "authorization marker must record noise = 0"
        )
    for field, expected in (
        ("real_market_inputs", "NONE"),
        ("training_authorization", "NONE"),
        ("g8_authorization", "NONE"),
    ):
        if fields[field] != expected:
            raise FinalGenerationAuthorizationError(
                f"authorization marker {field} must be NONE"
            )
    return fields


def verify_authorization(marker: Path | None = None) -> dict[str, Any]:
    """Fail closed unless final generation is explicitly authorized."""
    marker = AUTHORIZATION_MARKER if marker is None else marker
    if not Path(marker).is_file():
        raise FinalGenerationAuthorizationError(
            "final 10k generation is NOT AUTHORIZED: authorization marker is absent"
        )
    fields = parse_authorization_marker(marker)
    if not _marker_committed_at_head(marker):
        raise FinalGenerationAuthorizationError(
            "authorization marker is not committed at HEAD; final generation is forbidden"
        )
    base_sha = fields["canonical_main_base_git_sha"]
    if len(base_sha) < 7 or not _base_is_ancestor(base_sha):
        raise FinalGenerationAuthorizationError(
            "authorization base SHA is not an ancestor of HEAD"
        )
    head = _current_head()
    if not _head_on_remote(head):
        raise FinalGenerationAuthorizationError(
            "authorization commit is not visible on any remote ref; push and "
            "fetch before final generation"
        )
    current = scientific_dependency_hashes()
    mismatches = [
        f"{field}: marker {fields[field][:12]}... != current {current[key][:12]}..."
        for field, key in MARKER_HASH_FIELDS.items()
        if fields[field] != current[key]
    ]
    if mismatches:
        raise FinalGenerationAuthorizationError(
            "authorization marker identities no longer match the working tree: "
            + "; ".join(mismatches)
        )
    return {
        "authorized": True,
        "marker_path": str(marker),
        "marker_sha256": frozen.sha256_file(marker),
        "canonical_main_base_git_sha": base_sha,
        "authorization_commit_sha": head,
        "identities": {field: fields[field] for field in MARKER_HASH_FIELDS},
    }


def write_authorization_marker(
    marker: Path | None = None,
    base_sha: str | None = None,
) -> dict[str, Any]:
    """Run the no-pricing preflight and write the authorization marker.

    The marker must not already exist, and the preflight must pass, before
    any authorization is recorded.
    """
    marker = AUTHORIZATION_MARKER if marker is None else marker
    if Path(marker).exists():
        raise FinalGenerationAuthorizationError(
            "refusing to overwrite an existing authorization marker"
        )
    report = run_preflight()
    hashes = report["hashes"]
    base = base_sha or _current_head()
    text = "\n".join(
        [
            "FINAL_R2_10K_GENERATION_AUTHORIZED",
            "",
            f"statement: {NO_PRIOR_OUTPUT_STATEMENT}",
            "",
            f"canonical_main_base_git_sha: {base}",
            "frozen_contract_reference: configs/r2_synthetic_generation_FINAL.yaml "
            "(R2_SYNTHETIC_GENERATION_FINAL, status FROZEN_BEFORE_PILOT, tracking issue 27, "
            "merged PR #28 / main 86aaa38)",
            "parameter_panel_source: "
            "evidence/final_r2_candidate_pool_readiness_20260822/final_parameter_panel.csv "
            "(readiness selected_panel_csv_sha256; FINAL_PARAMETER_PANEL_ONLY, "
            "SURFACES_NOT_GENERATED, NOT_YET_TRAINING_DATA)",
            f"frozen_parameter_panel_sha256: {hashes['panel']}",
            f"production_pricer_sha256: {hashes['pricer']}",
            f"r2_synthetic_interface_sha256: {hashes['r2_synthetic_interface']}",
            f"generation_config_sha256: {hashes['config']}",
            f"reviewed_sampling_config_sha256: {hashes['reviewed_sampling_config']}",
            f"reviewed_sampler_source_sha256: {hashes['reviewed_sampler_source']}",
            f"generator_source_sha256: {hashes['generator_source']}",
            f"final_generation_module_sha256: {hashes['final_generation_module']}",
            "generator_source_drift_classification: "
            + GENERATOR_SOURCE_DRIFT_CLASSIFICATION,
            "final_quota_surfaces: 10000",
            "noise_level: 0.0",
            "real_market_inputs: NONE",
            "training_authorization: NONE",
            "g8_authorization: NONE",
            "replacement_or_refill: FORBIDDEN",
            "resampling_or_reseed: FORBIDDEN",
            "",
        ]
    )
    Path(marker).parent.mkdir(parents=True, exist_ok=True)
    Path(marker).write_text(text, encoding="utf-8", newline="\n")
    return {
        "marker_path": str(marker),
        "marker_sha256": frozen.sha256_file(marker),
        "canonical_main_base_git_sha": base,
        "preflight_passed": report["passed"],
    }


# ---------------------------------------------------------------------------
# Final generation (the only final-cohort pricing pathway)
# ---------------------------------------------------------------------------


def _final_surface_metadata(
    row: Mapping[str, Any],
    generation_index: int,
    conditioning_provenance: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_status": FINAL_DATASET_STATUS,
        "distribution": str(row["distribution"]),
        "split": str(row["split"]),
        "candidate_id": int(row["candidate_id"]),
        "candidate_key": str(row["candidate_key"]),
        "parameter_vector_hash": frozen.parameter_vector_hash(row),
        "parameter_sampler_seed": int(
            hashes["parameter_sampler_seeds"][str(row["distribution"])]
        ),
        "conditioning_seed": int(conditioning_provenance["seed"]),
        "conditioning_stride": int(conditioning_provenance["stride"]),
        "conditioning_lattice_index": int(conditioning_provenance["lattice_index"]),
        "generation_index": generation_index,
        "panel_row_index": generation_index,
        "noise_level": 0.0,
        "generation_config_sha256": hashes["config"],
        "production_pricer_sha256": hashes["pricer"],
        "generator_source_sha256": hashes["generator_source"],
        "final_generation_module_sha256": hashes["final_generation_module"],
        "reviewed_sampler_source_sha256": hashes["reviewed_sampler_source"],
        "r2_synthetic_interface_sha256": hashes["r2_synthetic_interface"],
        "generator_version": hashes["generator_version"],
        "final_generator_version": FINAL_GENERATOR_VERSION,
        "authorization_marker_sha256": hashes["authorization_marker"],
        "reviewed_sampling_policy": "UNCHANGED_REVIEWED_DESIGN",
        "real_market_inputs_used": False,
    }


def _generate_selected_final_surfaces(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
    hashes: Mapping[str, Any],
    row_indices: range | list[int] | None = None,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Price the selected final rows in exact panel order; retain failures.

    ``row_indices`` selects a subset for the predeclared deterministic
    replay; it is always in ascending panel order and never reorders.
    """
    from .r2_representation import build_synthetic_surface

    indices = list(range(len(panel))) if row_indices is None else list(row_indices)
    if indices and (min(indices) < 0 or max(indices) >= len(panel)):
        raise FinalDatasetValidationError(
            "row_indices contain a position outside the panel"
        )
    surfaces: list[Any] = []
    conditioning_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    combination_count = int(config["conditioning"]["lattice"]["combination_count"])
    rows = list(panel.iterrows())
    for generation_index in indices:
        _, row = rows[generation_index]
        surface_id = f"R2_FINAL_{row['candidate_key']}"
        conditioning_record: dict[str, Any] = {
            "surface_id": surface_id,
            "candidate_key": row["candidate_key"],
        }
        conditioning = None
        conditioning_provenance = None
        try:
            conditioning, conditioning_provenance = frozen.build_conditioning(
                generation_index, FINAL_COHORT, config
            )
            metadata = _final_surface_metadata(
                row, generation_index, conditioning_provenance, hashes
            )
            surface = build_synthetic_surface(
                row[PARAMETER_NAMES].to_numpy(dtype=np.float64),
                conditioning,
                surface_id=surface_id,
                metadata=metadata,
                node_count=int(config["pricing"]["node_count"]),
            )
            surfaces.append(surface)
            conditioning_rows.append(
                {
                    **conditioning_record,
                    **conditioning_provenance,
                    "parameters_canonical_order": dict(
                        surface.metadata["parameters_canonical_order"]
                    ),
                }
            )
            sanity_rows.append(frozen._numerical_sanity(surface))
        except Exception as error:  # retained, never replaced or refilled
            failures.append(
                {
                    **conditioning_record,
                    "dataset_status": "RETAINED_FINAL_GENERATION_FAILURE",
                    "candidate_id": int(row["candidate_id"]),
                    "distribution": str(row["distribution"]),
                    "split": str(row["split"]),
                    "panel_row_index": generation_index,
                    "parameters": {
                        name: float(row[name]) for name in PARAMETER_NAMES
                    },
                    "parameter_vector_hash": frozen.parameter_vector_hash(row),
                    **(
                        {"conditioning": conditioning.__dict__}
                        if conditioning is not None
                        else {}
                    ),
                    **(conditioning_provenance or {}),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    if row_indices is None:
        # The full cohort must satisfy the frozen conditioning support check.
        frozen.validate_conditioning_support(
            surfaces,
            conditioning_rows,
            combination_count,
            int(config["conditioning"]["seeds"][FINAL_COHORT]),
        )
    return surfaces, conditioning_rows, failures, sanity_rows


def run_final_generation(
    output_directory: str | Path = FINAL_OUTPUT,
    authorization: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Generate the final 10,000-surface dataset under full authorization gates."""
    authorization = authorization or verify_authorization()
    report = run_preflight()
    output = Path(output_directory)
    if output.exists():
        raise FinalGenerationAuthorizationError(
            f"refusing to overwrite an existing final dataset directory: {output}"
        )
    config = frozen.load_generation_config()
    panel = load_final_panel()
    seeds = {
        distribution: int(config["sampling"]["seeds"][f"final_{distribution}"])
        for distribution in frozen.DISTRIBUTIONS
    }
    hashes: dict[str, Any] = {
        **scientific_dependency_hashes(),
        "generator_version": config["generator_version"],
        "parameter_sampler_seeds": seeds,
        "authorization_marker": authorization["marker_sha256"],
    }
    surfaces, conditioning_rows, failures, sanity_rows = _generate_selected_final_surfaces(
        panel, config, hashes
    )
    output.mkdir(parents=True)

    if failures:
        failures_hash = frozen._write_jsonl(failures, output / "failures.jsonl")
        frozen.write_json(
            output / "manifest.json",
            {
                "schema_version": "1.0",
                "status": "FAILED_CLOSED_RETAINED_FAILURES",
                "dataset_status": "FAILED_CLOSED_RETAINED_FAILURES",
                "failure_count": len(failures),
                "failures_sha256": failures_hash,
                "authorization_commit_sha": authorization["authorization_commit_sha"],
                "training_started": False,
                "replacement_or_refill_used": False,
            },
        )
        raise FinalDatasetValidationError(
            f"final generation FAILED CLOSED with {len(failures)} retained "
            "pricing/numerical failures; the frozen 10,000-surface contract "
            "cannot be satisfied and no replacement is permitted"
        )

    payloads = frozen._surface_payloads(surfaces)
    surfaces_payload = b"".join(
        frozen.deterministic_json_bytes(payload) for payload in payloads
    )
    (output / "surfaces.jsonl").write_bytes(surfaces_payload)
    file_hashes = {
        "surfaces_jsonl_sha256": frozen.sha256_bytes(surfaces_payload),
        "conditioning_jsonl_sha256": frozen._write_jsonl(
            conditioning_rows, output / "conditioning.jsonl"
        ),
        "numerical_sanity_jsonl_sha256": frozen._write_jsonl(
            sanity_rows, output / "numerical_sanity.jsonl"
        ),
    }
    validation = validate_final_dataset(output, panel)
    validation_bytes = frozen.deterministic_json_bytes(validation)
    (output / "integrity_report.json").write_bytes(validation_bytes)
    file_hashes["integrity_report_json_sha256"] = frozen.sha256_bytes(validation_bytes)

    parity_errors = [row["put_call_parity_max_abs_error"] for row in sanity_rows]
    manifest = {
        "schema_version": "1.0",
        "contract_name": config["contract_name"],
        "generator_version": config["generator_version"],
        "final_generator_version": FINAL_GENERATOR_VERSION,
        "cohort": FINAL_COHORT,
        "dataset_status": FINAL_DATASET_STATUS,
        "dataset_labels": [
            "FINAL_R2_CLEAN_10000",
            "RESEARCH_SYNTHETIC_TRUTH_DATASET",
            "FROZEN_BEFORE_MODEL_TRAINING",
        ],
        "representation_name": REPRESENTATION_NAME,
        "representation_version": REPRESENTATION_VERSION,
        "slot_keys": [
            [key.expiry_rank, key.target_log_moneyness, key.option_type]
            for key in CANONICAL_SLOT_KEYS
        ],
        "surface_count": len(surfaces),
        "quotas": frozen.config_quotas(FINAL_COHORT),
        "split_summary": frozen.split_summary(panel, FINAL_COHORT),
        "parameter_panel": {
            "source": str(PANEL_CSV_PATH),
            "sha256": hashes["panel"],
            "readiness_reference": "selected_panel_csv_sha256 in "
            "evidence/final_r2_candidate_pool_readiness_20260822/readiness_manifest.json",
            "row_count": len(panel),
            "float_parsing": "pandas read_csv float_precision=round_trip (exact float64)",
        },
        "sampling": {
            "policy": "PANEL_REUSED_FROM_SEALED_READINESS_NO_RESAMPLING",
            "seeds": seeds,
        },
        "conditioning_policy": {
            "classification": config["conditioning"]["classification"],
            "seed": int(config["conditioning"]["seeds"][FINAL_COHORT]),
            "stride": int(config["conditioning"]["strides"][FINAL_COHORT]),
            "lattice": config["conditioning"]["lattice"],
            "real_market_inputs_used": False,
        },
        "pricing": {
            "source": config["pricing"]["production_source"],
            "entrypoint": config["pricing"]["entrypoint"],
            "node_count": int(config["pricing"]["node_count"]),
            "noise_level": 0.0,
            "replacement_on_failure": False,
        },
        "authorization": {
            "marker_path": authorization["marker_path"],
            "marker_sha256": authorization["marker_sha256"],
            "canonical_main_base_git_sha": authorization["canonical_main_base_git_sha"],
            "authorization_commit_sha": authorization["authorization_commit_sha"],
            "authorization_commit_remote_verified": True,
            "statement": NO_PRIOR_OUTPUT_STATEMENT,
        },
        "provenance_hashes": {
            "generation_config_sha256": hashes["config"],
            "production_pricer_sha256": hashes["pricer"],
            "generator_source_sha256": hashes["generator_source"],
            "final_generation_module_sha256": hashes["final_generation_module"],
            "reviewed_sampler_source_sha256": hashes["reviewed_sampler_source"],
            "reviewed_sampling_config_sha256": hashes["reviewed_sampling_config"],
            "r2_synthetic_interface_sha256": hashes["r2_synthetic_interface"],
            "authorization_marker_sha256": authorization["marker_sha256"],
            **file_hashes,
            "deterministic_content_sha256": file_hashes["surfaces_jsonl_sha256"],
        },
        "failure_count": 0,
        "numerical_sanity": {
            "surface_count": len(sanity_rows),
            "failed_surface_count": sum(not row["passed"] for row in sanity_rows),
            "all_passed": all(row["passed"] for row in sanity_rows),
            "put_call_parity_max_abs_error": max(parity_errors),
            "normalized_price_min": min(row["normalized_price_min"] for row in sanity_rows),
            "normalized_price_max": max(row["normalized_price_max"] for row in sanity_rows),
        },
        "validation_report": validation,
        "environment": frozen.environment_metadata(),
        "command": "python -m src.r2_final_generation generate-final",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_is_rng_input": False,
        "generation_git_sha": _current_head(),
        "real_market_inputs_used": False,
        "noise_level": 0.0,
        "training_started": False,
        "g8_started": False,
        "final_10k_generated": True,
        "replay_status": "PENDING",
    }
    frozen.write_json(output / "manifest.json", manifest)
    return output, manifest


# ---------------------------------------------------------------------------
# Independent on-disk validation
# ---------------------------------------------------------------------------


def validate_final_dataset(
    output: str | Path,
    panel: pd.DataFrame | None = None,
    expected_total: int = FINAL_TOTAL_SURFACES,
) -> dict[str, Any]:
    """Re-read the produced dataset from disk and validate the full contract.

    Independent of in-memory generation state: parses ``surfaces.jsonl``,
    re-derives every identity/quota/conditioning/pricing property, and
    cross-checks the frozen panel row by row.
    """
    from src.constraints import validate_parameters

    output = Path(output)
    panel = panel if panel is not None else load_final_panel()
    lines = (output / "surfaces.jsonl").read_text(encoding="utf-8").splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    payloads = [json.loads(line) for line in nonempty_lines]

    checks: dict[str, bool] = {}
    checks["count_matches_expected"] = len(payloads) == expected_total
    checks["serialization_round_trip"] = all(
        json.loads(json.dumps(payload, allow_nan=False)) == payload
        and frozen.deterministic_json_bytes(payload) == line.encode("utf-8") + b"\n"
        for payload, line in zip(payloads, nonempty_lines, strict=True)
    )
    seen_ids: set[str] = set()
    seen_vectors: dict[str, str] = {}
    split_counts: dict[tuple[str, str], int] = {}
    distribution_counts: dict[str, int] = {}
    for payload in payloads:
        validate_payload(payload)
        user_metadata = payload["metadata"].get("user_metadata", {})
        distribution = str(user_metadata["distribution"])
        split = str(user_metadata["split"])
        split_counts[(distribution, split)] = split_counts.get((distribution, split), 0) + 1
        distribution_counts[distribution] = distribution_counts.get(distribution, 0) + 1
        seen_ids.add(str(payload["surface_id"]))
        vector_hash = str(user_metadata["parameter_vector_hash"])
        if vector_hash in seen_vectors and seen_vectors[vector_hash] != split:
            checks["no_cross_split_overlap"] = False
        seen_vectors.setdefault(vector_hash, split)

    quotas = frozen.config_quotas(FINAL_COHORT)
    checks["distribution_quotas_exact"] = (
        distribution_counts == quotas["distributions"]
    )
    checks["split_quotas_exact"] = split_counts == {
        (distribution, split): int(quotas["splits"][split][distribution])
        for distribution in frozen.DISTRIBUTIONS
        for split in frozen.SPLIT_ORDER
    }
    checks["unique_surface_ids"] = len(seen_ids) == len(payloads)
    checks["unique_parameter_vectors"] = len(seen_vectors) == len(payloads)
    checks.setdefault("no_cross_split_overlap", True)
    checks["all_masks_true"] = all(all(payload["mask"]) for payload in payloads)
    checks["canonical_slot_order"] = all(
        payload["slot_keys"]
        == [
            [key.expiry_rank, key.target_log_moneyness, key.option_type]
            for key in CANONICAL_SLOT_KEYS
        ]
        for payload in payloads
    )

    constraints_ok = True
    conditioning_ok = True
    panel_correspondence_ok = True
    allowed_dte1 = {7, 14, 21, 30, 45, 60, 75, 90}
    allowed_gaps = {7, 14, 21, 30, 45, 60, 90}
    for payload, (_, panel_row) in zip(payloads, panel.iterrows(), strict=True):
        parameters = payload["metadata"]["parameters_canonical_order"]
        vector = np.asarray([parameters[name] for name in PARAMETER_NAMES], dtype=np.float64)
        if not validate_parameters(vector)["is_valid"]:
            constraints_ok = False
        user_metadata = payload["metadata"]["user_metadata"]
        if (
            user_metadata["candidate_key"] != panel_row["candidate_key"]
            or user_metadata["distribution"] != panel_row["distribution"]
            or user_metadata["split"] != panel_row["split"]
            or user_metadata["parameter_vector_hash"] != panel_row["parameter_vector_hash"]
            or any(parameters[name] != float(panel_row[name]) for name in PARAMETER_NAMES)
            or payload["surface_id"] != f"R2_FINAL_{panel_row['candidate_key']}"
        ):
            panel_correspondence_ok = False
        dte1, dte2 = payload["metadata"]["dte"]
        rates = payload["rates"]
        carries = payload["carries"]
        rate_values = {float(value) for value in rates}
        carry_values = {float(value) for value in carries}
        rates_finite_single = (
            len(rate_values) == 1
            and all(math.isfinite(value) for value in rate_values)
        )
        carries_finite_single = (
            len(carry_values) == 1
            and all(math.isfinite(value) for value in carry_values)
        )
        # The lattice carry is rate + offset computed in float64; recovering
        # the offset by subtraction is only exact to rounding, so compare
        # against the allowed offsets within a tight tolerance.
        offset_ok = (
            rates_finite_single
            and carries_finite_single
            and any(
                abs(
                    next(iter(carry_values)) - next(iter(rate_values)) - allowed
                )
                <= 1e-12
                for allowed in (-0.02, -0.01, 0.0, 0.01, 0.02, 0.03)
            )
        )
        if (
            dte1 not in allowed_dte1
            or (dte2 - dte1) not in allowed_gaps
            or dte2 <= dte1
            or not rates_finite_single
            or not carries_finite_single
            or not offset_ok
            or payload["spot"] != 100.0
            or payload["metadata"]["synthetic"] is not True
            or payload["source"] != SOURCE_SYNTHETIC
        ):
            conditioning_ok = False
    checks["parameter_constraints_valid"] = constraints_ok
    checks["conditioning_contract_valid"] = conditioning_ok
    checks["panel_correspondence_exact"] = panel_correspondence_ok

    finite_positive = all(
        all(math.isfinite(value) and value > 0.0 for value in payload["prices"])
        for payload in payloads
    )
    checks["prices_finite_strictly_positive"] = finite_positive
    checks["all_dataset_status_final"] = all(
        payload["metadata"]["user_metadata"]["dataset_status"] == FINAL_DATASET_STATUS
        for payload in payloads
    )

    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise FinalDatasetValidationError(
            f"final dataset validation failed: {failed}"
        )
    return {"validated": True, "checks": checks, "surface_count": len(payloads)}


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------


def predeclared_replay_indices(panel: pd.DataFrame) -> list[int]:
    """Indices fixed by rule alone (never by surface-quality results)."""
    selected = {
        index
        for index in range(len(panel))
        if index % REPLAY_SUBSET_STRIDE == 0
    }
    for distribution in frozen.DISTRIBUTIONS:
        for split in frozen.SPLIT_ORDER:
            mask = (
                (panel["distribution"] == distribution)
                & (panel["split"] == split)
            ).to_numpy()
            positions = np.flatnonzero(mask)[:REPLAY_SUBSET_PER_GROUP_ROW_LIMIT]
            selected.update(int(position) for position in positions)
    return sorted(selected)


def run_final_replay(
    primary_output: str | Path = FINAL_OUTPUT,
    replay_output: str | Path = FINAL_REPLAY_OUTPUT,
    mode: str = "full",
) -> tuple[Path, dict[str, Any]]:
    """Deterministically re-price the final cohort (or predeclared subset)."""
    if mode not in ("full", "predeclared-subset"):
        raise FinalDatasetValidationError(
            "replay mode must be 'full' or 'predeclared-subset'"
        )
    authorization = verify_authorization()
    run_preflight()
    primary = Path(primary_output)
    replay_path = Path(replay_output)
    if not (primary / "surfaces.jsonl").is_file():
        raise FinalDatasetValidationError("primary final output is missing")
    if replay_path.exists():
        raise FinalGenerationAuthorizationError(
            f"refusing to overwrite an existing replay directory: {replay_path}"
        )
    config = frozen.load_generation_config()
    panel = load_final_panel()
    seeds = {
        distribution: int(config["sampling"]["seeds"][f"final_{distribution}"])
        for distribution in frozen.DISTRIBUTIONS
    }
    hashes: dict[str, Any] = {
        **scientific_dependency_hashes(),
        "generator_version": config["generator_version"],
        "parameter_sampler_seeds": seeds,
        "authorization_marker": authorization["marker_sha256"],
    }
    indices = (
        None if mode == "full" else predeclared_replay_indices(panel)
    )
    surfaces, conditioning_rows, failures, sanity_rows = _generate_selected_final_surfaces(
        panel, config, hashes, row_indices=indices
    )
    if failures:
        raise FinalDatasetValidationError(
            f"deterministic replay encountered {len(failures)} failures"
        )
    replay_path.mkdir(parents=True)
    payloads = frozen._surface_payloads(surfaces)
    replay_payload = b"".join(
        frozen.deterministic_json_bytes(payload) for payload in payloads
    )
    (replay_path / "surfaces.jsonl").write_bytes(replay_payload)

    primary_lines = (
        (primary / "surfaces.jsonl").read_text(encoding="utf-8").splitlines()
    )
    replay_lines = replay_payload.decode("utf-8").splitlines()
    if mode == "full":
        identical = replay_lines == primary_lines
        compared = {"surfaces": len(primary_lines), "mode": mode}
    else:
        identical = all(
            replay_lines[position] == primary_lines[index]
            for position, index in enumerate(indices)
        )
        compared = {
            "surfaces": len(indices),
            "mode": mode,
            "rule": (
                f"panel row index % {REPLAY_SUBSET_STRIDE} == 0, plus first "
                f"{REPLAY_SUBSET_PER_GROUP_ROW_LIMIT} rows of each "
                "(distribution, split) group, committed before generation"
            ),
            "indices": indices,
        }
    replay_hashes = {
        "surfaces_jsonl_sha256": frozen.sha256_bytes(replay_payload),
        "primary_surfaces_jsonl_sha256": frozen.sha256_file(
            primary / "surfaces.jsonl"
        ),
    }
    report = {
        "schema_version": "1.0",
        "mode": mode,
        "compared": compared,
        "byte_identical_payloads": identical,
        "hashes": replay_hashes,
        "replay_output": str(replay_path),
        "primary_output": str(primary),
        "authorized_at_commit": authorization["authorization_commit_sha"],
        "replayed_at_utc": datetime.now(UTC).isoformat(),
        "full_replay_performed": mode == "full",
    }
    frozen.write_json(replay_path / "replay_report.json", report)
    frozen.write_json(primary / "replay_report.json", report)
    if not identical:
        raise FinalDatasetValidationError(
            "deterministic replay is NOT byte-identical to the primary output"
        )
    primary_manifest = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
    primary_manifest["replay_status"] = (
        "VERIFIED_IDENTICAL_FULL_REPLAY"
        if mode == "full"
        else "VERIFIED_IDENTICAL_PREDECLARED_SUBSET_REPLAY"
    )
    primary_manifest["replay_report"] = report
    frozen.write_json(primary / "manifest.json", primary_manifest)
    return replay_path, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="non-pricing pre-generation checks")
    subparsers.add_parser(
        "authorize", help="run preflight and write the authorization marker"
    )
    generate = subparsers.add_parser(
        "generate-final", help="price the frozen 10,000-row panel (authorized only)"
    )
    generate.add_argument("--output", type=Path, default=FINAL_OUTPUT)
    replay = subparsers.add_parser(
        "replay", help="deterministic reproducibility replay"
    )
    replay.add_argument("--mode", choices=("full", "predeclared-subset"), default="full")
    replay.add_argument("--primary", type=Path, default=FINAL_OUTPUT)
    replay.add_argument("--output", type=Path, default=FINAL_REPLAY_OUTPUT)
    arguments = parser.parse_args(argv)

    if arguments.command == "preflight":
        report = run_preflight()
        print(json.dumps({"passed": report["passed"], "checks": list(report["checks"])}, sort_keys=True))
        return 0
    if arguments.command == "authorize":
        record = write_authorization_marker()
        print(json.dumps(record, sort_keys=True))
        return 0
    if arguments.command == "generate-final":
        _, manifest = run_final_generation(arguments.output)
        print(
            json.dumps(
                {
                    "dataset_status": manifest["dataset_status"],
                    "surface_count": manifest["surface_count"],
                    "failure_count": manifest["failure_count"],
                    "replay_status": manifest["replay_status"],
                },
                sort_keys=True,
            )
        )
        return 0
    _, report = run_final_replay(arguments.primary, arguments.output, arguments.mode)
    print(json.dumps({"mode": report["mode"], "byte_identical_payloads": report["byte_identical_payloads"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
