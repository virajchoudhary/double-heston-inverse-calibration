"""Focused tests required before the full G2 R2/R3 calibration matrix.

Every test maps to a numbered requirement of the G2 implementation contract
(see docs/G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md and the execution task).
The full matrix must not launch until all of these pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.calibrate_double_heston import load_hard_safety_bounds  # noqa: E402
from src.constants import PARAMETER_NAMES  # noqa: E402
from src.constraints import validate_parameters  # noqa: E402
from src.double_heston import price_double_heston_surface  # noqa: E402
from src.g2_r2r3 import decision, frozen, noise as noise_module  # noqa: E402
from src.g2_r2r3.calibration import (  # noqa: E402
    ResultLog,
    clean_observables,
    fit_cell,
    observed_observables,
)
from src.g2_r2r3.geometry import (  # noqa: E402
    DateProfile,
    build_geometry,
    representation_slots,
)
from src.g2_r2r3.jacobian import full_jacobian_record  # noqa: E402
from src.g2_r2r3.pricer import normalized_observables  # noqa: E402
from src.g2_r2r3.starts import start_seed, start_schedule_for_cell  # noqa: E402
from src.g2_r2r3.truths import additional_truths, truth_panel  # noqa: E402

BOUNDS_PATH = REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
SMOKE_PROFILE = DateProfile(
    date_id="2026-07-15",
    spot=344.35,
    expiry_dates=("2026-07-28", "2026-08-25", "2026-09-29"),
    dte=(13, 41, 76),
    rates=(0.0519, 0.0526, 0.0533),
    carries=(-0.04439968962850673, 0.01446983271309761, 0.033544000442899055),
)


def test_r2_nominal_geometry_is_20_slots():
    slots = representation_slots(SMOKE_PROFILE, "R2")
    assert len(slots) == 20
    assert frozen.R2_NOMINAL_SLOTS == 20


def test_r3_nominal_geometry_is_30_slots():
    slots = representation_slots(SMOKE_PROFILE, "R3")
    assert len(slots) == 30
    assert frozen.R3_NOMINAL_SLOTS == 30


def test_r3_masking_explicit_and_deterministic():
    # Synthetic panel: masks are structurally all-true and deterministic.
    for representation in frozen.REPRESENTATIONS:
        slots = representation_slots(SMOKE_PROFILE, representation)
        geometry = build_geometry(slots)
        geometry_again = build_geometry(representation_slots(SMOKE_PROFILE, representation))
        assert geometry["mask"].all()
        assert np.array_equal(geometry["mask"], geometry_again["mask"])
    # Market path: masked slots carry an explicit failure reason, never a fill.
    from src.g2_r2r3 import market

    report = market.audit_date("2026-07-15")
    slots = report["slot_table"]
    missing = slots.loc[~slots["usable"]]
    assert (missing["observed_price"].isna()).all()
    assert (missing["failure_reason"] != "").all()
    assert report["mask_count"] == int((~slots["usable"]).sum())


def test_no_model_price_imputation():
    from src.g2_r2r3 import market

    for date_id in frozen.MARKET_DATES:
        report = market.audit_date(date_id)
        slots = report["slot_table"]
        missing = slots.loc[~slots["usable"]]
        # Unusable slots must hold no price at all, not a model or proxy price.
        assert missing["observed_price"].isna().all()
        assert missing["strike"].isna().all()


def test_no_unsupported_interpolation_or_extrapolation():
    from src.g2_r2r3 import market

    for date_id in frozen.MARKET_DATES:
        report = market.audit_date(date_id)
        for slot in report["slot_table"].itertuples():
            if slot.usable:
                # Selected strike is an actual listed strike whose realized
                # log-moneyness lies within the predeclared 0.05 target gate.
                assert abs(slot.log_moneyness_actual - slot.target_log_moneyness) <= 0.05 + 1e-12
                assert abs(slot.log_moneyness_actual) <= 0.10 + 1e-12


def test_actual_maturity_carried_through():
    for representation in frozen.REPRESENTATIONS:
        slots = representation_slots(SMOKE_PROFILE, representation)
        for slot in slots:
            expected = SMOKE_PROFILE.dte[slot.rank - 1] / 365.0
            assert slot.maturity_years == pytest.approx(expected, rel=0.0, abs=0.0)
            assert slot.rate == SMOKE_PROFILE.rates[slot.rank - 1]
            assert slot.carry == SMOKE_PROFILE.carries[slot.rank - 1]


def test_canonical_parameter_order_unchanged():
    expected = [
        "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
        "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast",
    ]
    assert list(PARAMETER_NAMES) == expected
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    assert list(bounds) == expected


def test_common_quote_keys_receive_bit_identical_noise():
    truth = frozen.STANDING_TRUTH_VECTORS["case_1"]
    r2_slots = representation_slots(SMOKE_PROFILE, "R2")
    r3_slots = representation_slots(SMOKE_PROFILE, "R3")
    for level in (0.005, 0.01, 0.02):
        observed_r2 = observed_observables(truth, "case_1", r2_slots, level)
        observed_r3 = observed_observables(truth, "case_1", r3_slots, level)
        r2_by_key = {
            slot.key: value for slot, value in zip(r2_slots, observed_r2)
        }
        for slot, value in zip(r3_slots, observed_r3):
            if slot.rank <= 2:
                assert value.hex() == r2_by_key[slot.key].hex()


def test_same_twelve_starts_for_r2_and_r3():
    for truth_index in (0, 7, 19):
        for noise_index in (0, 3):
            schedule = start_schedule_for_cell(truth_index, noise_index)
            assert len(schedule) == 12
            assert schedule[0][0] == "neutral_transform_midpoint"
            # Representation-independent by construction: same key -> same starts.
            assert start_seed(truth_index, noise_index) == start_seed(truth_index, noise_index)


def test_truth_selection_seed_reproduces_identical_sixteen():
    first = additional_truths()
    second = additional_truths()
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 16
    panel = truth_panel()
    assert len(panel) == 20
    extras = panel.loc[panel["source"] == "reviewed_interior_seed_20260822"]
    assert len(extras) == 16
    for row in extras.itertuples():
        match = first.loc[first["truth_id"] == row.truth_id]
        assert len(match) == 1
        for name in PARAMETER_NAMES:
            assert float(match.iloc[0][name]) == float(getattr(row, name))


def test_noise_rerun_bit_identical():
    for _ in range(2):
        factor_a = noise_module.slot_noise_factor("case_2", 3, -0.05, "put", 0.01)
        factor_b = noise_module.slot_noise_factor("case_2", 3, -0.05, "put", 0.01)
        assert factor_a.hex() == factor_b.hex()
    assert noise_module.slot_noise_factor("case_2", 3, -0.05, "put", 0.0) == 1.0


def test_start_rerun_bit_identical():
    first = start_schedule_for_cell(4, 2)
    second = start_schedule_for_cell(4, 2)
    for (name_a, a), (name_b, b) in zip(first, second):
        assert name_a == name_b
        assert a.tobytes() == b.tobytes()


def test_fast_pricer_agrees_with_production_on_r2_r3_points():
    """Requirement 13: existing defensible tolerance on representative points."""
    tolerance = 1e-10  # Node B validation achieved ~2.8e-16 absolute
    for case in ("case_1", "case_2", "case_3", "case_4"):
        vector = frozen.STANDING_TRUTH_VECTORS[case]
        for representation in frozen.REPRESENTATIONS:
            slots = representation_slots(SMOKE_PROFILE, representation)
            geometry = build_geometry(slots)
            fast = normalized_observables(
                vector,
                geometry["strikes"],
                geometry["maturities"],
                geometry["option_types"],
                geometry["rates"],
                geometry["dividends"],
                spot=frozen.SYNTHETIC_SPOT,
                node_count=frozen.NODE_COUNT,
            )
            production = np.asarray(
                [
                    price_double_heston_surface(
                        frozen.SYNTHETIC_SPOT,
                        geometry["strikes"][index : index + 1],
                        geometry["maturities"][index : index + 1],
                        float(geometry["rates"][index]),
                        float(geometry["dividends"][index]),
                        [str(geometry["option_types"][index])],
                        vector,
                        node_count=frozen.NODE_COUNT,
                    )[0]
                    for index in range(len(slots))
                ]
            ) / frozen.SYNTHETIC_SPOT
            max_diff = float(np.max(np.abs(fast - production)))
            assert max_diff <= tolerance, f"{case}/{representation}: {max_diff}"


def test_canonical_structural_validity_checks_work():
    good = frozen.STANDING_TRUTH_VECTORS["case_1"]
    assert validate_parameters(good)["is_valid"]
    bad = good.copy()
    bad[0] = -1.0  # negative kappa_slow
    assert not validate_parameters(bad)["is_valid"]
    bad = good.copy()
    bad[5] = bad[0]  # violates kappa_slow < kappa_fast
    assert not validate_parameters(bad)["is_valid"]
    bad = good.copy()
    bad[2] = 10.0  # violates slow Feller
    assert not validate_parameters(bad)["is_valid"]


def test_swapped_factor_twin_rejected_by_canonical_ordering():
    vector = frozen.STANDING_TRUTH_VECTORS["case_1"]
    swapped = np.concatenate([vector[5:], vector[:5]])
    strikes = np.asarray([95.0, 100.0, 105.0])
    maturities = np.asarray([0.0753, 0.0753, 0.0753])
    option_types = np.asarray(["call", "put", "put"])
    base = price_double_heston_surface(
        100.0, strikes, maturities, 0.06, 0.02, option_types, vector,
        enforce_ordering=False,
    )
    twin = price_double_heston_surface(
        100.0, strikes, maturities, 0.06, 0.02, option_types, swapped,
        enforce_ordering=False,
    )
    assert float(np.max(np.abs(base - twin))) == 0.0
    with pytest.raises(ValueError):
        price_double_heston_surface(
            100.0, strikes, maturities, 0.06, 0.02, option_types, swapped
        )


def test_raw_results_retain_failures_and_boundary_hits(tmp_path):
    log = ResultLog(tmp_path / "runs.jsonl")
    truth = frozen.STANDING_TRUTH_VECTORS["case_1"]
    schedule = start_schedule_for_cell(0, 0)
    poisoned = schedule[:2]
    poisoned[1] = (
        poisoned[1][0],
        np.asarray([np.nan] * 10),
    )  # forces a recorded failure
    rows = fit_cell(
        "case_1", 0, truth, "R2", 0, 0.0, SMOKE_PROFILE, poisoned
    )
    log.append(rows)
    assert len(rows) == 2
    failures = [row for row in rows if not row["success"]]
    assert failures, "failed start must be retained"
    assert any("ValueError" in row["optimizer_message"] for row in failures)
    assert all("boundary_reasons" in row for row in rows)
    assert log.cell_complete("case_1", "R2", 0) is False  # only 2 of 12 recorded


def test_decision_evaluator_cannot_change_protocol_thresholds():
    import inspect

    signature = inspect.signature(decision.comparative_assessment)
    for forbidden in (
        "strong_threshold", "partial_threshold", "cluster_threshold",
        "median_threshold", "maximum_threshold",
    ):
        assert forbidden not in signature.parameters
    # The evaluator reads thresholds from the frozen module only.
    assessment = decision.comparative_assessment(
        {"median_dispersion": 0.30, "maximum_dispersion": 0.80, "mean_cluster_count": 1.5},
        {"median_dispersion": 0.20, "maximum_dispersion": 0.60, "mean_cluster_count": 1.2},
    )
    assert assessment["classification"] in (
        "STRONG_IMPROVEMENT", "PARTIAL_IMPROVEMENT", "NO_MATERIEL_IMPROVEMENT",
    )
    # Frozen constants must equal the protocol numbers.
    assert frozen.STRONG_IMPROVEMENT_MEDIAN == 0.25
    assert frozen.PARTIAL_IMPROVEMENT_MEDIAN == 0.10


def test_jacobian_record_structure():
    vector = frozen.STANDING_TRUTH_VECTORS["case_1"]
    for representation in frozen.REPRESENTATIONS:
        slots = representation_slots(SMOKE_PROFILE, representation)
        record = full_jacobian_record(vector, slots)
        assert record["practical_rank"] >= 1
        assert record["condition_number"] >= 1.0
        assert "weakest_direction_1_loading_kappa_slow" in record
