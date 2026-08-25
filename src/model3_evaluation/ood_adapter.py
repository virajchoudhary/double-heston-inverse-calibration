"""Validation-only future interface to the separately frozen OOD benchmark."""

from __future__ import annotations

from typing import Any, Mapping


OOD_REQUEST_SCHEMA = "MODEL3_OOD_EVALUATION_REQUEST_V1"
OOD_PROTOCOL_CONFIG_SHA256 = (
    "948a23e7d30f762d9d6d85bff79c5c83c51624e943b4f1f5dd94e7038a348e7c"
)
OOD_COHORT_CONTENT_SHA256 = (
    "e8b117ac93f6319e634fa28d6dd5ed884e86e130cf420e65eb8ef8da0276b7e4"
)


class OODAdapterError(ValueError):
    """A proposed OOD intake lacks the separately frozen benchmark contract."""


def validate_ood_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate future Model3/OOD intake without exposing or computing outcomes."""
    required = frozenset(
        {
            "schema", "three_seed_freeze_manifest_sha256", "ood_protocol_git_sha",
            "ood_cohort_manifest_sha256", "cohort_generation_owned_by_model3_layer",
            "ood_protocol_config_sha256", "ood_cohort_content_sha256",
            "checkpoint_identity_required", "row_surface_alignment_required",
            "model3_tuning_allowed", "result_intake_schema",
        }
    )
    if set(payload) != required:
        raise OODAdapterError("OOD-request schema mismatch")
    if payload["schema"] != OOD_REQUEST_SCHEMA:
        raise OODAdapterError("wrong OOD-request schema")
    if payload["result_intake_schema"] != "MODEL3_OOD_RESULT_INTAKE_V1":
        raise OODAdapterError("unknown OOD result-intake schema")
    if payload["ood_protocol_config_sha256"] != OOD_PROTOCOL_CONFIG_SHA256:
        raise OODAdapterError("OOD protocol config identity mismatch")
    if payload["ood_cohort_content_sha256"] != OOD_COHORT_CONTENT_SHA256:
        raise OODAdapterError("OOD frozen-cohort content identity mismatch")
    for field in (
        "three_seed_freeze_manifest_sha256", "ood_protocol_git_sha",
        "ood_cohort_manifest_sha256", "ood_protocol_config_sha256",
        "ood_cohort_content_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
            raise OODAdapterError(f"{field} must be a lowercase SHA-256 identity")
    requirements = {
        "cohort_generation_owned_by_model3_layer": False,
        "checkpoint_identity_required": True,
        "row_surface_alignment_required": True,
        "model3_tuning_allowed": False,
    }
    mismatches = [
        key for key, value in requirements.items() if payload.get(key) is not value
    ]
    if mismatches:
        raise OODAdapterError(f"OOD isolation requirement mismatch: {mismatches}")
    return {
        "accepted_for_future_intake": True,
        "executed_by_this_adapter": False,
        "protocol_owner": "SEPARATELY_FROZEN_OOD_BRANCH",
    }
