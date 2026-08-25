"""Validation-only future G8 interface; never updates Model3 weights."""

from __future__ import annotations

from typing import Any, Mapping


G8_REQUEST_SCHEMA = "MODEL3_G8_EVALUATION_REQUEST_V1"
G8_INCLUSION_CONDITION = "FROZEN_AND_COMMITTED_BEFORE_G8_ACQUISITION"


class G8AdapterError(ValueError):
    """A proposed G8 intake would violate evaluation-only isolation."""


def validate_g8_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a future G8 observation-intake request without execution."""
    required = frozenset(
        {
            "schema", "g8_protocol_git_sha", "g8_result_intake_schema",
            "real_market_weight_updates_allowed", "training_allowed",
            "checkpoint_selection_allowed", "hyperparameter_tuning_allowed",
            "pricing_family_and_inverse_method_comparisons_separate",
            "frozen_checkpoint_identity_required", "model3_inclusion_condition",
        }
    )
    if set(payload) != required:
        raise G8AdapterError("G8-request schema mismatch")
    if payload["schema"] != G8_REQUEST_SCHEMA:
        raise G8AdapterError("wrong G8-request schema")
    if payload["g8_result_intake_schema"] != "MODEL3_G8_RESULT_INTAKE_V1":
        raise G8AdapterError("unknown G8 result-intake schema")
    protocol_sha = payload["g8_protocol_git_sha"]
    if not isinstance(protocol_sha, str) or len(protocol_sha) != 40:
        raise G8AdapterError("g8_protocol_git_sha must be a Git SHA")
    forbidden_true = {
        "real_market_weight_updates_allowed", "training_allowed",
        "checkpoint_selection_allowed", "hyperparameter_tuning_allowed",
    }
    nonfalse = [
        key for key in forbidden_true
        if not isinstance(payload.get(key), bool) or payload[key] is not False
    ]
    if nonfalse:
        raise G8AdapterError(f"G8 forbidden fields must be boolean false: {nonfalse}")
    if payload.get("pricing_family_and_inverse_method_comparisons_separate") is not True:
        raise G8AdapterError("pricing-family and inverse-method comparisons must stay separate")
    if payload.get("frozen_checkpoint_identity_required") is not True:
        raise G8AdapterError("G8 intake requires frozen checkpoint identity")
    if payload["model3_inclusion_condition"] != G8_INCLUSION_CONDITION:
        raise G8AdapterError("Model3 does not satisfy the G8 inclusion condition")
    return {
        "accepted_for_future_intake": True,
        "executed_by_this_adapter": False,
        "weight_update_quarantine": "ENFORCED",
    }
