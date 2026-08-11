from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_g2_identifiability_analysis as baseline
import scripts.run_g2_multi_date_identifiability as multi
from src.calibrate_double_heston import load_hard_safety_bounds


VALID_PARAMETERS = np.asarray(
    [1.2, 0.04, 0.25, -0.35, 0.03, 3.0, 0.02, 0.25, -0.25, 0.02],
    dtype=np.float64,
)

PREVIOUS_G2_HASHES = {
    "docs/G2_COMMON_SUPPORT_ANALYSIS.md": "7DC923AE288579443B2E998A9BA88CDD7B337CD599C0F5133EBD44874D3C5227",
    "docs/G2_IDENTIFIABILITY_ANALYSIS.md": "4CC75351EEA503FC5386AECCE329803AAA54A1AE9FD7D9A1248D6249BDA87CC3",
    "docs/G2_INFORMATION_REMEDIATION.md": "AFDE07B6E67326040B7996AF7B39613434D8565B3A466F1225C1616BC7656F53",
    "scripts/run_g2_common_support_analysis.py": "24B3AD8B74B639D4931E695DB8C108595678FFEFB22832EF82B211783CC8E3AE",
    "scripts/run_g2_identifiability_analysis.py": "70354017001EDBFBBF7215937316E251FA2CADDF7E8FC921C9DEE137BC7F5ECD",
    "scripts/run_g2_information_remediation.py": "3D02EF7FDBEA84F79A5021A39E521E597883DF50C193705465AEF9A8EF39CA09",
    "tests/test_g2_common_support_analysis.py": "361FDB6387A33C13014FBFD543578CB70CE62E0D6BCF77A9548EB0F39BE5B6F1",
    "tests/test_g2_identifiability_analysis.py": "71542785FB45EBCE3B7408779E6483C7D0A709BF87DDFF936020C07E57E0BB60",
    "tests/test_g2_information_remediation.py": "7C90CC851217F57FF5B6E7F3A9FE5E9372FA77600C518710CAB35736AC2212F5",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _one_sample() -> pd.DataFrame:
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    return baseline.select_representative_parameters(
        bounds, per_distribution=4
    ).head(1)


def test_exact_three_date_design_and_canonical_target_contract() -> None:
    assert multi.VALUATION_DATES == (
        "2026-07-01",
        "2026-07-15",
        "2026-07-22",
    )
    assert multi.DATE_GAPS_DAYS == (14, 7)
    assert multi.MATURITY_PROFILES == (
        ("2026-07-01", (27, 55)),
        ("2026-07-15", (13, 41)),
        ("2026-07-22", (6, 34)),
    )
    assert multi.CANONICAL_TARGET_NAMES == tuple(baseline.PARAMETER_NAMES)
    assert len(multi.CANONICAL_TARGET_NAMES) == 10
    assert multi.SHARED_STRUCTURAL_NAMES == (
        "kappa_slow",
        "theta_slow",
        "sigma_slow",
        "rho_slow",
        "kappa_fast",
        "theta_fast",
        "sigma_fast",
        "rho_fast",
    )
    matrix = multi.experiment_designs().set_index("design_id")
    assert list(matrix.index) == ["A", "B", "C", "D"]
    assert matrix.loc["A", "normalized_price_count"] == 20
    assert matrix.loc["B", "normalized_price_count"] == 60
    assert matrix.loc["C", "nuisance_state_count"] == 4
    assert matrix.loc["D", "exact_cir_transition_density"]
    assert multi.RECOVERY_MAXITER == 80
    assert multi.RECOVERY_SAMPLES_PER_DISTRIBUTION == 1


def test_exact_cir_transition_is_deterministic_and_not_conditional_mean() -> None:
    values = [
        multi.exact_cir_transition_from_uniform(
            1.2, 0.04, 0.25, 0.03, 14.0 / 365.0, 0.731
        )
        for _ in range(2)
    ]
    assert values[0] == values[1]
    conditional_mean = 0.04 + (0.03 - 0.04) * math.exp(-1.2 * 14.0 / 365.0)
    assert values[0] != conditional_mean
    assert math.isfinite(
        multi.exact_cir_transition_logpdf(
            values[0], 1.2, 0.04, 0.25, 0.03, 14.0 / 365.0
        )
    )


def test_simulated_later_states_are_date_specific_and_replay_exactly() -> None:
    samples = _one_sample()
    first = multi.simulate_state_paths(samples)
    second = multi.simulate_state_paths(samples)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    row = first.iloc[0]
    assert row["v_slow_t1"] != row["v_slow_t0"]
    assert row["v_fast_t1"] != row["v_fast_t0"]
    assert row["v_slow_t2"] != row["v_slow_t1"]
    assert row["v_fast_t2"] != row["v_fast_t1"]
    # Exact stochastic states are not forced back into the anchor-v0 envelope.
    assert multi.STATE_BOUNDS["v_slow_t1"][0] < 0.005
    for name in multi.NUISANCE_STATE_NAMES:
        assert multi.STATE_BOUNDS[name][0] < row[name] < multi.STATE_BOUNDS[name][1]


def test_nuisance_transform_has_target_blind_state_centers() -> None:
    bounds = load_hard_safety_bounds(baseline.BOUNDS_PATH)
    _, states = multi._decode_latent(
        np.zeros(14, dtype=np.float64), multi.DESIGN_BY_ID["C"], bounds
    )
    assert states is not None
    np.testing.assert_allclose(states, multi.STATE_START_CENTERS, rtol=0.0, atol=1.0e-15)


def test_no_constant_v0_shortcut_and_state_effect_is_date_local() -> None:
    states = np.asarray([0.031, 0.021, 0.032, 0.022], dtype=np.float64)
    changed = states.copy()
    changed[0] += 0.01
    design = multi.DESIGN_BY_ID["C"]
    base_prices = multi.joint_normalized_prices(
        VALID_PARAMETERS,
        design,
        oracle_states=None,
        nuisance_states=states,
        node_count=8,
    )
    changed_prices = multi.joint_normalized_prices(
        VALID_PARAMETERS,
        design,
        oracle_states=None,
        nuisance_states=changed,
        node_count=8,
    )
    np.testing.assert_array_equal(base_prices[:20], changed_prices[:20])
    assert not np.array_equal(base_prices[20:40], changed_prices[20:40])
    np.testing.assert_array_equal(base_prices[40:], changed_prices[40:])


def test_nuisance_projection_removes_only_nuisance_column_space() -> None:
    nuisance = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=float
    )
    target = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    projected, rank = multi.nuisance_projected_jacobian(target, nuisance)
    assert rank == 2
    np.testing.assert_allclose(projected[:, :2], 0.0, atol=1.0e-15)
    np.testing.assert_array_equal(projected[:, 2], target[:, 2])


def test_coupled_noise_reuses_anchor_date_coordinates() -> None:
    single, joint = multi.coupled_noise(123, 0.005)
    assert single.shape == (20,)
    assert joint.shape == (60,)
    np.testing.assert_array_equal(single, joint[:20])


def test_previous_stage_a_and_g2_files_match_protected_hashes() -> None:
    snapshot = multi._protected_snapshot()
    for relative, expected in PREVIOUS_G2_HASHES.items():
        assert snapshot[relative] == expected
        assert _digest(multi.REPOSITORY_ROOT / relative) == expected


def test_quick_analysis_replays_deterministically(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_report = tmp_path / "first.md"
    second_report = tmp_path / "second.md"
    first = multi.run_analysis(
        output_root=first_root,
        report_path=first_report,
        node_count=8,
        sample_limit=1,
        skip_recovery=True,
    )
    second = multi.run_analysis(
        output_root=second_root,
        report_path=second_report,
        node_count=8,
        sample_limit=1,
        skip_recovery=True,
    )
    assert first["artifact_hashes"] == second["artifact_hashes"]
    assert first_report.read_bytes() == second_report.read_bytes()
    for relative in first["artifact_hashes"]:
        assert (first_root / relative).read_bytes() == (second_root / relative).read_bytes()
