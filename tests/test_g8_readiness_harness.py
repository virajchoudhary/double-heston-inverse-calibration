from __future__ import annotations

import numpy as np
import pytest

from src.calibrate_double_heston import load_hard_safety_bounds
from src.g8_readiness.checkpoints import stage_canonical_checkpoint
from src.g8_readiness.harness import real_g8_traditional_starts


def test_real_traditional_adapter_has_exactly_two_uninformed_starts() -> None:
    starts = real_g8_traditional_starts(load_hard_safety_bounds("configs/parameter_bounds_PROVISIONAL.yaml"))
    assert [name for name, _value in starts] == [
        "neutral_transform_midpoint", "deterministic_broad_start"
    ]
    assert np.array_equal(starts[0][1], np.zeros(10))
    assert np.isfinite(starts[1][1]).all()
    assert all(name != "disclosed_target_perturbation" for name, _value in starts)


def test_external_checkpoint_staging_refuses_hash_mismatch(tmp_path):
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"canonical-bytes")
    expected = {
        "method": "MODEL1_ANN", "seed": 11,
        "path": "checkpoints/forbidden/test.pt", "sha256": "0" * 64,
    }
    with pytest.raises(Exception, match="SHA mismatch"):
        stage_canonical_checkpoint(expected, source_path=source, approve_expected_hash=True)
    with pytest.raises(Exception, match="hash approval"):
        stage_canonical_checkpoint(expected, source_path=source, approve_expected_hash=False)
