"""Filtering exposed-case assessments must preserve seeded observations."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assess_regular_pinn_filters", ROOT / "scripts/mentor_dh_pinn/assess_regular_pinn.py")
assessment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assessment)


def test_condition_filters_preserve_seeded_truth_observations_and_development_label(tmp_path, monkeypatch):
    # This tests sampling/filtering and artifact semantics, not pricing accuracy.
    # Deterministic Black prices and a fixed fake fit avoid MLX/training work.
    checkpoint = tmp_path / "fake_checkpoint.bin"
    checkpoint.write_bytes(b"test-only-frozen-checkpoint")
    info = {"label": "test_network", "checkpoint": str(checkpoint),
            "sha256": assessment.sha256(checkpoint), "config": {"factors": 2}}
    monkeypatch.setattr(assessment, "load_checkpoint", lambda path: (SimpleNamespace(factors=2), info))

    def exact(x, tau, unit, factors, nodes=128):
        return assessment.black_call(x, (0.4 + 0.1 * unit[0])**2 * tau)

    def fit(model, x, tau, observed_iv, **kwargs):
        unit = np.full(10, 0.5)
        return {"status": "fitted", "unit": unit.tolist(),
                "physical": assessment.decode_unit(unit, 2).tolist(),
                "seconds": 0.0, "optimizer_success": True, "total_nfev": 1, "starts": []}

    monkeypatch.setattr(assessment, "_exact", exact)
    monkeypatch.setattr(assessment, "fit_network", fit)
    monkeypatch.setattr(assessment, "_network_iv", lambda model, x, tau, unit: np.full(len(x), 0.4 + 0.1 * unit[0]))

    def run(name, *filters):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["assess_regular_pinn.py", "--checkpoint", str(checkpoint),
                                          "--out", str(output), "--seed", "73121", "--cases", "3",
                                          "--starts", "1", "--max-nfev", "1", *filters])
        assessment.main()
        with (output / "observations.csv").open(newline="") as stream:
            observations = list(csv.DictReader(stream))
        return (json.loads((output / "synthetic_truth.json").read_text()), observations,
                json.loads((output / "manifest.json").read_text()))

    full_truth, full_observations, full_manifest = run("full")
    assert len(full_truth) == 3
    assert len(full_observations) == 3 * (63 + 126) * 2
    assert full_manifest["evidence_level"] == "frozen assessment"
    for level in ("0", "0.01"):
        truth, observations, manifest = run(f"rich_noise_{level}", "--geometry", "rich",
                                            "--noise", level, "--development")
        selected = [row for row in full_observations
                    if row["geometry"] == "rich" and float(row["noise"]) == float(level)]
        assert truth == full_truth
        assert observations == selected  # Exact string equality, including noisy prices and IVs.
        assert len(observations) == 3 * 126
        assert manifest["status"] == "complete"
        assert manifest["evidence_level"] == "development on exposed cases"
        assert manifest["evaluated_geometries"] == ["rich"]
        assert manifest["evaluated_noise_levels"] == [float(level)]
        assert manifest["frozen_checkpoint_and_source_hashes_rechecked"] is True
