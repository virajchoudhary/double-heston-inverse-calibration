from __future__ import annotations

from datetime import date

import pytest

from src.g8_readiness.state_machine import assess_pre_acquisition_freeze
from src.g8_readiness.contracts import canonical_slot_roles


READY_INPUTS = {
    "protocol_frozen": True,
    "protocol_identity_verified": True,
    "config_identity_verified": True,
    "tool_identities_verified": True,
    "checkpoint_gate_passed": True,
    "independent_review_verdict": "APPROVED",
    "model3_decision": {"label": "MODEL3_NOT_YET_ELIGIBLE_FOR_G8_INCLUSION"},
    "current_date": date(2027, 1, 1),
}


def _assess(**changes):
    values = dict(READY_INPUTS)
    values.update(changes)
    return assess_pre_acquisition_freeze(**values)


def test_ready_state_never_grants_explicit_acquisition() -> None:
    result = _assess()
    assert result["status"] == "READY_NOW"
    assert result["ready_for_pre_acquisition_freeze"] is True
    assert result["explicit_acquisition_authorized"] is False


def test_checkpoint_and_review_are_ordered_prerequisites() -> None:
    assert _assess(protocol_identity_verified=False)[
        "status"
    ] == "WAITING_FOR_PROTOCOL_IDENTITY_VERIFICATION"
    assert _assess(config_identity_verified=False)[
        "status"
    ] == "WAITING_FOR_CONFIG_IDENTITY_VERIFICATION"
    assert _assess(tool_identities_verified=False)[
        "status"
    ] == "WAITING_FOR_TOOL_IDENTITY_VERIFICATION"
    assert _assess(checkpoint_gate_passed=False)["status"] == "WAITING_FOR_CHECKPOINT"
    assert _assess(independent_review_verdict="REVIEW_INCOMPLETE_TIMEOUT")[
        "status"
    ] == "WAITING_FOR_INDEPENDENT_REVIEW"


def test_model3_and_calendar_do_not_become_execution_authorization() -> None:
    assert _assess(model3_decision={"label": "INVALID"})[
        "status"
    ] == "WAITING_FOR_MODEL3_FREEZE_DECISION"
    assert _assess(current_date=date(2026, 9, 29))["status"] == "WAITING_FOR_DATE_FLOOR"
    assert _assess(model3_decision={"label": "MODEL3_INCLUDED"})[
        "status"
    ] == "WAITING_FOR_MODEL3_FREEZE_DECISION"


def test_nominal_pricing_roles_are_disjoint() -> None:
    roles = canonical_slot_roles()
    calibration = set(roles["pricing_family_calibration"].nonzero()[0])
    holdout = set(roles["pricing_family_holdout"].nonzero()[0])
    assert len(calibration) == 12 and len(holdout) == 8
    assert calibration.isdisjoint(holdout)
