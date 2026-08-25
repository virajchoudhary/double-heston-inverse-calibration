"""Validation-only future intake interface for the frozen Issue #34 noise protocol."""

from __future__ import annotations

from typing import Any, Mapping


NOISE_REQUEST_SCHEMA = "MODEL3_ISSUE34_NOISE_EVALUATION_REQUEST_V1"
FROZEN_NOISE_PROTOCOL_CONFIG_SHA256 = (
    "2fa49b3eb885d3427c01ab0cfe447fc6ddd7f19957db73c4b4ed782476c57c5a"
)


class NoiseAdapterError(ValueError):
    """A proposed noise evaluation lacks a required frozen prerequisite."""


def validate_noise_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the future request without reading or generating cohort outcomes."""
    required = frozenset(
        {
            "schema", "three_seed_freeze_manifest_sha256",
            "clean_evaluation_manifest_sha256", "clean_evaluation_completion_state",
            "issue34_protocol_git_sha", "issue34_cohort_manifest_sha256",
            "issue34_protocol_config_sha256",
            "paired_cohort_identity_required", "model3_tuning_allowed",
            "noisy_observation_repricing_required", "clean_latent_repricing_required",
            "parameter_recovery_required", "degradation_curves_required",
        }
    )
    if set(payload) != required:
        raise NoiseAdapterError("noise-request schema mismatch")
    if payload["schema"] != NOISE_REQUEST_SCHEMA:
        raise NoiseAdapterError("wrong noise-request schema")
    for field in (
        "three_seed_freeze_manifest_sha256", "clean_evaluation_manifest_sha256",
        "issue34_protocol_git_sha", "issue34_cohort_manifest_sha256",
        "issue34_protocol_config_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
            raise NoiseAdapterError(f"{field} must be a lowercase SHA-256 identity")
    if payload["clean_evaluation_completion_state"] != "COMPLETE":
        raise NoiseAdapterError("clean Model3 evaluation must already be COMPLETE")
    if payload["issue34_protocol_config_sha256"] != FROZEN_NOISE_PROTOCOL_CONFIG_SHA256:
        raise NoiseAdapterError("Issue #34 protocol config identity mismatch")
    boolean_requirements = {
        "paired_cohort_identity_required": True,
        "model3_tuning_allowed": False,
        "noisy_observation_repricing_required": True,
        "clean_latent_repricing_required": True,
        "parameter_recovery_required": True,
        "degradation_curves_required": True,
    }
    mismatches = [
        key for key, value in boolean_requirements.items() if payload.get(key) is not value
    ]
    if mismatches:
        raise NoiseAdapterError(f"noise isolation requirement mismatch: {mismatches}")
    return {
        "accepted_for_future_execution": True,
        "executed_by_this_adapter": False,
        "cohort_generation": "REUSE_FROZEN_ISSUE34_IDENTITIES_ONLY",
    }
