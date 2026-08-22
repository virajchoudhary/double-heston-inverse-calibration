"""Focused tests for the canonical frozen-R2 representation interface.

Each test names the numbered requirement it covers from the Issue #25
milestone contract (20 required checks), plus contract-hardening cases from
the adversarial review questions A-E:

A. synthetic and real surfaces share identical slot meaning;
B. a masked quote can never enter numerics as an observation;
C. legacy 108-input data can never silently pass as R2;
D. slot ordering cannot change without a test failing;
E. serialization cannot lose maturity/rate/carry/mask information.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.calibrate_double_heston import load_hard_safety_bounds  # noqa: E402
from src.constants import PARAMETER_NAMES  # noqa: E402
from src.constraints import validate_parameters  # noqa: E402
from src.double_heston import price_double_heston_surface  # noqa: E402
from src.g2_r2r3 import frozen as g2_frozen  # noqa: E402
from src.g2_r2r3.geometry import DateProfile, representation_slots  # noqa: E402
from src.r2_representation import (  # noqa: E402
    CANONICAL_SLOT_KEYS,
    CENTRAL_FIVE_LOG_MONEYNESS,
    LEGACY_108_INPUT_SIZE,
    MASKED_PRICE_PLACEHOLDER,
    NOMINAL_SLOT_COUNT,
    R2Conditioning,
    R2Surface,
    REJECTED_R3_INPUT_SIZE,
    REPRESENTATION_NAME,
    REPRESENTATION_VERSION,
    RealSurfaceNotConstructibleError,
    RepresentationContractError,
    SlotKey,
    build_real_surface,
    build_synthetic_surface,
    canonical_slot_keys,
    dataset_manifest,
    manifest_from_payload,
    payload_to_surface,
    read_surface_json,
    surface_from_vectors,
    surface_to_payload,
    validate_payload,
    write_surface_json,
)

BOUNDS_PATH = REPOSITORY_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml"
PACKAGE_SOURCES = sorted((REPOSITORY_ROOT / "src" / "r2_representation").glob("*.py"))


@lru_cache(maxsize=None)
def _audit_report(date_id: str) -> dict:
    from src.g2_r2r3 import market

    return market.audit_date(date_id)


DEV_PROFILE_0715 = DateProfile(
    date_id="2026-07-15",
    spot=344.35,
    expiry_dates=("2026-07-28", "2026-08-25", "2026-09-29"),
    dte=(13, 41, 76),
    rates=(0.0519, 0.0526, 0.0533),
    carries=(-0.04439968962850673, 0.01446983271309761, 0.033544000442899055),
)
DEV_CONDITIONING = R2Conditioning.from_date_profile(DEV_PROFILE_0715)


def _synthetic_surface() -> R2Surface:
    return build_synthetic_surface(
        g2_frozen.STANDING_TRUTH_VECTORS["case_1"],
        DEV_CONDITIONING,
        surface_id="test_synth_case_1",
    )


def _mask_everywhere(usable: list[bool]) -> list[bool]:
    return usable


# -- 1-5: slot identity and ordering ------------------------------------------


def test_requirement_1_exactly_20_nominal_slots():
    assert NOMINAL_SLOT_COUNT == 20
    assert len(CANONICAL_SLOT_KEYS) == 20
    surface = _synthetic_surface()
    assert surface.slot_count == 20
    assert len(surface.prices) == len(surface.mask) == len(surface.maturities) == 20


def test_requirement_2_slot_key_order_deterministic():
    assert canonical_slot_keys() == canonical_slot_keys()
    assert CANONICAL_SLOT_KEYS == canonical_slot_keys()
    # D: the canonical order is exactly the reviewed G2 study ordering.
    g2_keys = tuple(slot.key for slot in representation_slots(DEV_PROFILE_0715, "R2"))
    assert CANONICAL_SLOT_KEYS == g2_keys
    # The interface constants mirror the sealed frozen study constants.
    assert CENTRAL_FIVE_LOG_MONEYNESS == g2_frozen.CENTRAL_FIVE
    assert len(CANONICAL_SLOT_KEYS) == g2_frozen.R2_NOMINAL_SLOTS


def test_requirement_3_five_moneyness_values_exact():
    for option_type in ("call", "put"):
        for rank in (1, 2):
            block = [
                key.target_log_moneyness
                for key in CANONICAL_SLOT_KEYS
                if key.option_type == option_type and key.expiry_rank == rank
            ]
            assert block == list(CENTRAL_FIVE_LOG_MONEYNESS)
    assert CENTRAL_FIVE_LOG_MONEYNESS == (-0.10, -0.05, 0.0, 0.05, 0.10)


def test_requirement_4_exactly_two_expiry_ranks():
    ranks = {key.expiry_rank for key in CANONICAL_SLOT_KEYS}
    assert ranks == {1, 2}
    surface = _synthetic_surface()
    assert {key.expiry_rank for key in surface.slot_keys} == {1, 2}


def test_requirement_5_calls_and_puts_both_represented():
    calls = [key for key in CANONICAL_SLOT_KEYS if key.option_type == "call"]
    puts = [key for key in CANONICAL_SLOT_KEYS if key.option_type == "put"]
    assert len(calls) == 10 and len(puts) == 10
    assert [key.option_type for key in CANONICAL_SLOT_KEYS] == ["call"] * 10 + ["put"] * 10
    assert CANONICAL_SLOT_KEYS[0] == SlotKey(1, -0.10, "call")
    assert CANONICAL_SLOT_KEYS[-1] == SlotKey(2, 0.10, "put")


# -- 6-8: mask semantics -------------------------------------------------------


def test_requirement_6_synthetic_mask_all_true():
    surface = _synthetic_surface()
    assert all(surface.mask)
    assert surface.usable_slot_count() == 20
    assert surface.masked_slot_keys() == ()
    assert all(price > 0.0 for price in surface.prices)


def test_requirement_7_real_unsupported_quote_masked_false():
    surface = build_real_surface("2026-07-01", audit_report=_audit_report("2026-07-01"))
    masked = surface.masked_slot_keys()
    # The sealed G2 audit recorded exactly 9 masked R2 slots on 2026-07-01.
    assert len(masked) == 9
    assert surface.usable_slot_count() == 11
    for key in masked:
        index = surface.slot_keys.index(key)
        assert surface.mask[index] is False
        assert surface.prices[index] == MASKED_PRICE_PLACEHOLDER
    reasons = surface.metadata["provenance"]["failure_reasons"]
    assert all(reasons[index] != "" for index, valid in enumerate(surface.mask) if not valid)
    assert all(reasons[index] == "" for index, valid in enumerate(surface.mask) if valid)


def test_requirement_8_no_model_price_imputation():
    real = build_real_surface("2026-07-01", audit_report=_audit_report("2026-07-01"))
    masked_indices = [index for index, valid in enumerate(real.mask) if not valid]
    assert masked_indices
    for index in masked_indices:
        # B: masked positions are exactly 0.0, never a model/proxy number.
        assert real.prices[index] == 0.0
        assert real.metadata["provenance"]["observed_raw_prices"][index] is None
        assert real.metadata["provenance"]["actual_strikes"][index] is None
    assert real.metadata["imputation"] == "NONE_MASKED_EXPLICITLY"
    # A surface that tries to smuggle a value into a masked slot is rejected.
    prices = list(real.prices)
    prices[masked_indices[0]] = 0.017  # a plausible-looking model price
    with pytest.raises(RepresentationContractError, match="masked slot"):
        surface_from_vectors(
            prices, real.mask, real.maturities, real.rates, real.carries,
            spot=real.spot, surface_id="bad", source=real.source,
        )
    # NaN is never a valid representation of missingness or of a price.
    prices[masked_indices[0]] = float("nan")
    with pytest.raises(RepresentationContractError, match="finite"):
        surface_from_vectors(
            prices, real.mask, real.maturities, real.rates, real.carries,
            spot=real.spot, surface_id="bad", source=real.source,
        )


# -- 9-12: conditioning preservation ------------------------------------------


def test_requirement_9_actual_maturity_preserved():
    synthetic = _synthetic_surface()
    for key, maturity in zip(synthetic.slot_keys, synthetic.maturities, strict=True):
        assert maturity == DEV_PROFILE_0715.dte[key.expiry_rank - 1] / 365.0
    real = build_real_surface("2026-07-15", audit_report=_audit_report("2026-07-15"))
    details = sorted(_audit_report("2026-07-15")["expiry_details"], key=lambda item: item["rank"])[:2]
    for key, maturity in zip(real.slot_keys, real.maturities, strict=True):
        assert maturity == details[key.expiry_rank - 1]["dte"] / 365.0


def test_requirement_10_rates_preserved():
    synthetic = _synthetic_surface()
    for key, rate in zip(synthetic.slot_keys, synthetic.rates, strict=True):
        assert rate == DEV_PROFILE_0715.rates[key.expiry_rank - 1]
    real = build_real_surface("2026-07-15", audit_report=_audit_report("2026-07-15"))
    details = sorted(_audit_report("2026-07-15")["expiry_details"], key=lambda item: item["rank"])[:2]
    for key, rate in zip(real.slot_keys, real.rates, strict=True):
        assert rate == details[key.expiry_rank - 1]["rate"]


def test_requirement_11_carry_preserved():
    synthetic = _synthetic_surface()
    for key, carry in zip(synthetic.slot_keys, synthetic.carries, strict=True):
        assert carry == DEV_PROFILE_0715.carries[key.expiry_rank - 1]
    real = build_real_surface("2026-07-15", audit_report=_audit_report("2026-07-15"))
    details = sorted(_audit_report("2026-07-15")["expiry_details"], key=lambda item: item["rank"])[:2]
    for key, carry in zip(real.slot_keys, real.carries, strict=True):
        assert carry == details[key.expiry_rank - 1]["carry"]


def test_requirement_12_spot_normalization_round_trip():
    synthetic = _synthetic_surface()
    denormalized = synthetic.denormalized_prices_array()
    # Synthetic normalized prices reproduce the production prices/spot to the
    # float64 unit round-off (prices are stored as price/spot).
    reproduced = np.asarray(
        _production_rank_prices(synthetic), dtype=np.float64
    )
    np.testing.assert_allclose(
        denormalized, reproduced, rtol=1e-13, atol=0.0,
        err_msg="synthetic normalized prices must round-trip through spot"
    )
    real = build_real_surface("2026-07-15", audit_report=_audit_report("2026-07-15"))
    raw = real.metadata["provenance"]["observed_raw_prices"]
    mask = real.mask_array()
    np.testing.assert_allclose(
        real.denormalized_prices_array()[mask],
        np.asarray([value for value, valid in zip(raw, real.mask) if valid], dtype=np.float64),
        rtol=1e-13, atol=0.0,
        err_msg="real normalized prices must round-trip through spot",
    )


def _production_rank_prices(surface: R2Surface) -> list[float]:
    """Direct production-pricer prices at the canonical target geometry."""
    vector = g2_frozen.STANDING_TRUTH_VECTORS["case_1"]
    out: list[float] = [0.0] * surface.slot_count
    for rank in (1, 2):
        indices = [i for i, key in enumerate(surface.slot_keys) if key.expiry_rank == rank]
        keys = [surface.slot_keys[i] for i in indices]
        strikes = [surface.spot * float(np.exp(key.target_log_moneyness)) for key in keys]
        prices = price_double_heston_surface(
            surface.spot,
            strikes,
            [surface.maturities[i] for i in indices],
            surface.rates[indices[0]],
            surface.carries[indices[0]],
            [key.option_type for key in keys],
            vector,
        )
        for position, index in enumerate(indices):
            out[index] = float(prices[position])
    return out


# -- 13-15: serialization and rejection rules ----------------------------------


def test_requirement_13_serialization_round_trip_lossless():
    synthetic = _synthetic_surface()
    payload = surface_to_payload(synthetic)
    assert payload["representation_name"] == REPRESENTATION_NAME
    assert payload["representation_version"] == REPRESENTATION_VERSION
    restored = payload_to_surface(payload)
    # E: every contract field survives bit-identically.
    assert restored.prices == synthetic.prices
    assert restored.mask == synthetic.mask
    assert restored.maturities == synthetic.maturities
    assert restored.rates == synthetic.rates
    assert restored.carries == synthetic.carries
    assert restored.spot == synthetic.spot
    assert restored.slot_keys == synthetic.slot_keys
    assert restored.surface_id == synthetic.surface_id
    assert restored.source == synthetic.source
    assert restored.metadata == synthetic.metadata
    # File round-trip produces identical bytes and an identical surface.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "surface.json"
        write_surface_json(synthetic, path)
        first_bytes = path.read_bytes()
        from_file = read_surface_json(path)
        assert from_file.prices == synthetic.prices
        assert from_file.metadata == synthetic.metadata
        write_surface_json(from_file, path)
        assert path.read_bytes() == first_bytes  # deterministic serialization


def test_requirement_14_invalid_slot_ordering_rejected():
    keys = list(CANONICAL_SLOT_KEYS)
    keys[0], keys[1] = keys[1], keys[0]
    surface = _synthetic_surface()
    with pytest.raises(RepresentationContractError, match="ordering violates the canonical"):
        surface_from_vectors(
            surface.prices, surface.mask, surface.maturities, surface.rates,
            surface.carries, spot=surface.spot, surface_id="bad", source="x",
            slot_keys=keys,
        )
    # put-block-first (a plausible legacy reordering) is likewise rejected.
    reordered = tuple(key for key in CANONICAL_SLOT_KEYS if key.option_type == "put") + tuple(
        key for key in CANONICAL_SLOT_KEYS if key.option_type == "call"
    )
    with pytest.raises(RepresentationContractError, match="position 0"):
        surface_from_vectors(
            surface.prices, surface.mask, surface.maturities, surface.rates,
            surface.carries, spot=surface.spot, surface_id="bad", source="x",
            slot_keys=reordered,
        )


def test_requirement_15_mismatched_vector_lengths_rejected():
    surface = _synthetic_surface()
    for length in (19, 21, REJECTED_R3_INPUT_SIZE, LEGACY_108_INPUT_SIZE):
        with pytest.raises(RepresentationContractError):
            surface_from_vectors(
                list(surface.prices) + [0.01] * (length - 20) if length > 20 else list(surface.prices)[:length],
                list(surface.mask) + [True] * (length - 20) if length > 20 else list(surface.mask)[:length],
                list(surface.maturities) + [0.1] * (length - 20) if length > 20 else list(surface.maturities)[:length],
                list(surface.rates) + [0.05] * (length - 20) if length > 20 else list(surface.rates)[:length],
                list(surface.carries) + [0.0] * (length - 20) if length > 20 else list(surface.carries)[:length],
                spot=surface.spot, surface_id="bad", source="x",
            )
    with pytest.raises(RepresentationContractError):
        surface_from_vectors(
            surface.prices, surface.mask, surface.maturities, surface.rates,
            surface.carries[:19], spot=surface.spot, surface_id="bad", source="x",
        )


# -- 16-18: frozen scientific contracts and legacy rejection -------------------


def test_requirement_16_canonical_parameter_contract_unchanged():
    expected = [
        "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
        "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast",
    ]
    assert list(PARAMETER_NAMES) == expected
    bounds = load_hard_safety_bounds(BOUNDS_PATH)
    assert list(bounds) == expected
    assert validate_parameters(g2_frozen.STANDING_TRUTH_VECTORS["case_1"])["is_valid"]
    # The synthetic constructor rejects constraint-violating parameters.
    bad = g2_frozen.STANDING_TRUTH_VECTORS["case_1"].copy()
    bad[0] = -1.0
    with pytest.raises(RepresentationContractError, match="constraints"):
        build_synthetic_surface(bad, DEV_CONDITIONING, surface_id="bad")


def test_requirement_17_production_pricer_unchanged():
    surface = _synthetic_surface()
    direct = _production_rank_prices(surface)
    np.testing.assert_allclose(
        surface.prices_array() * surface.spot,
        np.asarray(direct, dtype=np.float64),
        rtol=1e-13, atol=0.0,
        err_msg="the constructor must use the unchanged production pricer",
    )
    # The canonical factor-swap ordering convention still rejects the twin.
    vector = g2_frozen.STANDING_TRUTH_VECTORS["case_1"]
    swapped = np.concatenate([vector[5:], vector[:5]])
    with pytest.raises(ValueError):
        price_double_heston_surface(
            100.0, [100.0], [30.0 / 365.0], 0.05, 0.0, ["call"], swapped,
        )


def test_requirement_18_legacy_108_never_accepted_as_canonical_r2():
    surface = _synthetic_surface()
    # C: a 108-length price vector is rejected with an explicit legacy message.
    with pytest.raises(RepresentationContractError, match="rejected legacy 108-input grid"):
        surface_from_vectors(
            [0.01] * LEGACY_108_INPUT_SIZE, [True] * LEGACY_108_INPUT_SIZE,
            [0.1] * LEGACY_108_INPUT_SIZE, [0.05] * LEGACY_108_INPUT_SIZE,
            [0.0] * LEGACY_108_INPUT_SIZE, spot=100.0, surface_id="legacy", source="x",
        )
    # A payload claiming 108 slot keys is rejected at the schema boundary.
    payload = surface_to_payload(surface)
    legacy_payload = dict(payload)
    legacy_payload["slot_keys"] = [[1, value, "call"] for value in np.linspace(-0.3, 0.3, 9)] * 12
    with pytest.raises(RepresentationContractError, match="108"):
        validate_payload(legacy_payload)
    # A 30-slot (rejected R3) vector is also refused.
    with pytest.raises(RepresentationContractError, match="rejected R3"):
        surface_from_vectors(
            [0.01] * REJECTED_R3_INPUT_SIZE, [True] * REJECTED_R3_INPUT_SIZE,
            [0.1] * REJECTED_R3_INPUT_SIZE, [0.05] * REJECTED_R3_INPUT_SIZE,
            [0.0] * REJECTED_R3_INPUT_SIZE, spot=100.0, surface_id="r3", source="x",
        )


# -- 19-20: integration smokes -------------------------------------------------


def test_requirement_19_real_ntpc_development_date_integration_smoke():
    for date_id, expected_usable in (("2026-07-01", 11), ("2026-07-15", 19)):
        surface = build_real_surface(date_id, audit_report=_audit_report(date_id))
        assert surface.usable_slot_count() == expected_usable
        assert surface.slot_keys == CANONICAL_SLOT_KEYS
        assert surface.spot == pytest.approx(_audit_report(date_id)["spot"])
        assert surface.metadata["development_date_excluded_from_g8"] is True
        assert surface.metadata["date_id"] == date_id
        assert surface.metadata["imputation"] == "NONE_MASKED_EXPLICITLY"
        restored = payload_to_surface(surface_to_payload(surface))
        assert restored.prices == surface.prices
    # Dates outside the sealed development panel are refused outright.
    with pytest.raises(RealSurfaceNotConstructibleError, match="development dates"):
        build_real_surface("2026-10-01")


def test_requirement_20_synthetic_integration_smoke():
    surface = _synthetic_surface()
    assert surface.mask == (True,) * 20
    assert all(np.isfinite(surface.prices_array()))
    assert all(value > 0.0 for value in surface.prices)
    assert surface.metadata["synthetic"] is True
    assert surface.metadata["pricing_engine"] == "production_double_heston_unchanged"
    assert surface.metadata["imputation"] == "NONE_COMPLETE_BY_CONSTRUCTION"
    # A: synthetic and real surfaces share identical slot meaning/order.
    real = build_real_surface("2026-07-15", audit_report=_audit_report("2026-07-15"))
    assert surface.slot_keys == real.slot_keys
    # Deterministic reconstruction.
    again = build_synthetic_surface(
        g2_frozen.STANDING_TRUTH_VECTORS["case_1"], DEV_CONDITIONING,
        surface_id="test_synth_case_1",
    )
    assert again.prices == surface.prices
    # Dataset manifest round-trip over mixed synthetic/real surfaces.
    manifest = dataset_manifest([surface, real])
    restored = manifest_from_payload(manifest)
    assert [item.surface_id for item in restored] == [surface.surface_id, real.surface_id]
    assert restored[0].prices == surface.prices
    assert restored[1].mask == real.mask
    json.dumps(manifest, allow_nan=False)  # fully JSON-safe


# -- adversarial hardening (A-E) -------------------------------------------------


def test_adversarial_a_synthetic_and_real_same_slot_meaning():
    synthetic = _synthetic_surface()
    for date_id in ("2026-07-01", "2026-07-15"):
        real = build_real_surface(date_id, audit_report=_audit_report(date_id))
        assert synthetic.slot_keys == real.slot_keys == CANONICAL_SLOT_KEYS
        # Conditioning arrays are rank-indexed identically in both paths.
        for key in real.slot_keys:
            index = real.slot_keys.index(key)
            assert real.maturities[index] > 0.0
            assert real.rates[index] != 0.0 or real.carries[index] != 0.0


def test_adversarial_b_masked_quote_cannot_become_an_observation():
    real = build_real_surface("2026-07-01", audit_report=_audit_report("2026-07-01"))
    assert real.masked_slot_keys()
    # valid_prices_array() physically excludes masked positions.
    assert len(real.valid_prices_array()) == real.usable_slot_count()
    assert real.valid_prices_array().min() > 0.0
    # Masked slots can never be resurrected through serialization.
    payload = surface_to_payload(real)
    assert all(
        payload["prices"][index] == 0.0
        for index, valid in enumerate(payload["mask"])
        if not valid
    )


def test_adversarial_c_payload_schema_drift_rejected():
    payload = surface_to_payload(_synthetic_surface())
    wrong_name = dict(payload, representation_name="FROZEN_R3_RANKED_THREE_EXPIRY")
    with pytest.raises(RepresentationContractError, match="not the canonical R2"):
        validate_payload(wrong_name)
    wrong_version = dict(payload, representation_version="2.0")
    with pytest.raises(RepresentationContractError, match="version"):
        validate_payload(wrong_version)
    missing = {key: value for key, value in payload.items() if key != "mask"}
    with pytest.raises(RepresentationContractError, match="missing fields"):
        validate_payload(missing)
    non_bool_mask = dict(payload, mask=[1 if value else 0 for value in payload["mask"]])
    with pytest.raises(RepresentationContractError, match="boolean"):
        validate_payload(non_bool_mask)
    nan_price = dict(payload, prices=list(payload["prices"]))
    nan_price["prices"][0] = float("nan")
    with pytest.raises(RepresentationContractError, match="finite"):
        validate_payload(nan_price)


def test_adversarial_d_slot_order_locked_by_test():
    # D: the exact 20-key sequence is pinned literally; ANY change to identity
    # or order fails here even if constructors were edited to match.
    expected = [
        (1, -0.10, "call"), (1, -0.05, "call"), (1, 0.00, "call"), (1, 0.05, "call"), (1, 0.10, "call"),
        (2, -0.10, "call"), (2, -0.05, "call"), (2, 0.00, "call"), (2, 0.05, "call"), (2, 0.10, "call"),
        (1, -0.10, "put"), (1, -0.05, "put"), (1, 0.00, "put"), (1, 0.05, "put"), (1, 0.10, "put"),
        (2, -0.10, "put"), (2, -0.05, "put"), (2, 0.00, "put"), (2, 0.05, "put"), (2, 0.10, "put"),
    ]
    assert list(CANONICAL_SLOT_KEYS) == [SlotKey(*item) for item in expected]


def test_adversarial_e_serialization_preserves_all_conditioning():
    real = build_real_surface("2026-07-01", audit_report=_audit_report("2026-07-01"))
    restored = payload_to_surface(surface_to_payload(real))
    assert restored.maturities == real.maturities
    assert restored.rates == real.rates
    assert restored.carries == real.carries
    assert restored.mask == real.mask
    assert restored.spot == real.spot
    assert restored.metadata["provenance"] == real.metadata["provenance"]
    assert restored.metadata["dte"] == real.metadata["dte"]
    assert restored.metadata["rate_simple_yield"] == real.metadata["rate_simple_yield"]


def test_per_rank_conditioning_consistency_enforced():
    surface = _synthetic_surface()
    rates = list(surface.rates)
    rates[3] = rates[3] + 0.001  # same rank, different conditioning
    with pytest.raises(RepresentationContractError, match="constant within expiry rank"):
        surface_from_vectors(
            surface.prices, surface.mask, surface.maturities, rates, surface.carries,
            spot=surface.spot, surface_id="bad", source="x",
        )
    maturities = list(surface.maturities)
    for index, key in enumerate(surface.slot_keys):  # rank 2 entirely before rank 1
        if key.expiry_rank == 2:
            maturities[index] = maturities[index] - 0.1
    with pytest.raises(RepresentationContractError, match="chronological"):
        surface_from_vectors(
            surface.prices, surface.mask, maturities, surface.rates, surface.carries,
            spot=surface.spot, surface_id="bad", source="x",
        )


def test_conditioning_validation_and_g2_bridge():
    with pytest.raises(RepresentationContractError, match="strictly increasing"):
        R2Conditioning(
            date_id="d", spot=100.0, expiry_dates=("2026-08-25", "2026-07-28"),
            dte=(55, 27), rates=(0.05, 0.05), carries=(0.0, 0.0),
        )
    with pytest.raises(RepresentationContractError, match="exactly 2 values"):
        R2Conditioning(
            date_id="d", spot=100.0, expiry_dates=("2026-07-28",),
            dte=(27,), rates=(0.05,), carries=(0.0,),
        )
    # The G2 bridge takes the FIRST TWO eligible ranks and normalizes at 100.
    conditioning = R2Conditioning.from_date_profile(DEV_PROFILE_0715)
    assert conditioning.dte == (13, 41)
    assert conditioning.spot == g2_frozen.SYNTHETIC_SPOT == 100.0
    assert conditioning.maturities == (13 / 365.0, 41 / 365.0)


def test_interface_has_no_legacy_grid_dependency():
    # The primary R2 data contract must not depend on the rejected 108 grid.
    forbidden = ("LOG_MONEYNESS_GRID", "MATURITY_DAYS_GRID", "surface_grid", "expected_input_size")
    for path in PACKAGE_SOURCES:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} references legacy grid token {token}"


# -- adversarial-review regression tests ----------------------------------------


def test_review_real_constructor_rejects_mismatched_audit_report():
    # Supplying another date's audit report must not silently mislabel the
    # surface's valuation-date provenance.
    with pytest.raises(RealSurfaceNotConstructibleError, match="not the"):
        build_real_surface("2026-07-01", audit_report=_audit_report("2026-07-15"))


def test_review_nan_metadata_rejected_on_read_path(tmp_path):
    import tempfile

    surface = _synthetic_surface()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "surface.json"
        write_surface_json(surface, path)
        text = path.read_text(encoding="utf-8")
    poisoned = text.replace('"synthetic": true', '"synthetic": NaN')
    bad_file = tmp_path / "poisoned.json"
    bad_file.write_text(poisoned, encoding="utf-8")
    with pytest.raises(RepresentationContractError, match="NaN"):
        read_surface_json(bad_file)
    # The in-memory boundary rejects non-finite metadata values too.
    payload = surface_to_payload(surface)
    nan_payload = dict(payload, metadata=dict(payload["metadata"], bad=float("nan")))
    with pytest.raises(RepresentationContractError, match="serializable"):
        validate_payload(nan_payload)


def test_review_non_json_metadata_rejected_with_contract_error():
    surface = _synthetic_surface()
    metadata = dict(surface.metadata)
    metadata["bad_numpy"] = np.int64(3)  # not JSON-serializable
    with pytest.raises(RepresentationContractError, match="serializable"):
        surface_to_payload(
            surface_from_vectors(
                surface.prices, surface.mask, surface.maturities, surface.rates,
                surface.carries, spot=surface.spot, surface_id="x",
                source=surface.source, metadata=metadata,
            )
        )


def test_review_manifest_reader_validates_top_level_keys_and_ids():
    surface_a = _synthetic_surface()
    surface_b = build_real_surface("2026-07-15", audit_report=_audit_report("2026-07-15"))
    manifest = dataset_manifest([surface_a, surface_b])
    # Tampered manifest-level slot keys are rejected on read.
    tampered = dict(manifest, slot_keys=[[9, 9.9, "call"]] * 20)
    with pytest.raises(RepresentationContractError):
        manifest_from_payload(tampered)
    # Duplicate surface_ids are rejected on read.
    duplicated = dict(manifest, surfaces=[manifest["surfaces"][0], manifest["surfaces"][0]])
    with pytest.raises(RepresentationContractError, match="duplicate surface_ids"):
        manifest_from_payload(duplicated)


def test_review_numpy_slot_keys_rejected_with_contract_error():
    with pytest.raises(RepresentationContractError, match="not comparable"):
        surface_from_vectors(
            [0.01] * 20, [True] * 20, [0.1] * 20, [0.05] * 20, [0.0] * 20,
            spot=100.0, surface_id="x", source="x",
            slot_keys=np.zeros((20, 3)),
        )
