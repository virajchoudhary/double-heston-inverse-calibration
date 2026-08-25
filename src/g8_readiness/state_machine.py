"""Explicit pre-acquisition workflow gates without crossing execution locks."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

from .contracts import DATE_FLOOR



class G8StateTransitionError(ValueError):
    pass


def assess_pre_acquisition_freeze(
    *,
    protocol_frozen: bool,
    protocol_identity_verified: bool,
    config_identity_verified: bool,
    tool_identities_verified: bool,
    checkpoint_gate_passed: bool,
    independent_review_verdict: str | None,
    model3_decision: Mapping[str, Any],
    current_date: date,
) -> dict[str, str | bool | list[str]]:
    """Return the single earliest unresolved prerequisite, never an execution grant."""
    waiting: list[str] = []
    if not protocol_frozen:
        waiting.append("WAITING_FOR_PROTOCOL_IDENTITY")
    if not protocol_identity_verified:
        waiting.append("WAITING_FOR_PROTOCOL_IDENTITY_VERIFICATION")
    if not config_identity_verified:
        waiting.append("WAITING_FOR_CONFIG_IDENTITY_VERIFICATION")
    if not tool_identities_verified:
        waiting.append("WAITING_FOR_TOOL_IDENTITY_VERIFICATION")
    if not checkpoint_gate_passed:
        waiting.append("WAITING_FOR_CHECKPOINT")
    if independent_review_verdict != "APPROVED":
        waiting.append("WAITING_FOR_INDEPENDENT_REVIEW")
    label = model3_decision.get("label")
    model3_evidence_bound = (
        label in {"MODEL3_NOT_FROZEN_NOT_EVALUATED", "MODEL3_NOT_YET_ELIGIBLE_FOR_G8_INCLUSION"}
        or (
            label == "MODEL3_INCLUDED"
            and model3_decision.get("decision") == "MODEL3_INCLUDED"
            and isinstance(model3_decision.get("checks"), Mapping)
            and all(model3_decision["checks"].values())
        )
    )
    if not model3_evidence_bound:
        waiting.append("WAITING_FOR_MODEL3_FREEZE_DECISION")
    if current_date < DATE_FLOOR:
        waiting.append("WAITING_FOR_DATE_FLOOR")
    if waiting:
        return {
            "status": waiting[0],
            "ready_for_pre_acquisition_freeze": False,
            "all_outstanding_prerequisites": waiting,
            "explicit_acquisition_authorized": False,
        }
    return {
        "status": "READY_NOW",
        "ready_for_pre_acquisition_freeze": True,
        "all_outstanding_prerequisites": [],
        "explicit_acquisition_authorized": False,
    }
