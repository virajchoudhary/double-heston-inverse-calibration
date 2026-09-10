"""An unchanged warm start can legitimately remain the selected L-BFGS model."""
import hashlib
import json

import pytest

from scripts.mentor_dh_pinn.report_regular_pinn import collect_run


def test_iteration_zero_uses_its_checkpoint_and_effective_lbfgs_metadata(tmp_path):
    run = tmp_path / "lbfgs_run"
    data = tmp_path / "source_labels"
    run.mkdir()
    data.mkdir()

    def write(path, value):
        path.write_text(json.dumps(value))

    anchor_indices = list(range(16384))
    write(data / "manifest.json", {
        "splits": {"train": {"candidates": 131072, "usable": 131072},
                   "validation": {"candidates": 16384, "usable": 16384}},
        "collocation_points": 18000,
    })
    write(run / "config.json", {
        "factors": 2, "data": str(data), "skip_recovery": False,
        "seed": 908021, "resume": "parent/model.safetensors",
        "weight_decay": 1e-6, "lr": .001, "steps": 12000,
        "resample_every": 1000, "anchors": 131072,
    })
    write(run / "effective_optimizer_settings.json", {
        "effective_config": {
            "optimizer": "L-BFGS-B", "weight_decay": 0., "resample_every": 0,
            "sensitivity": .2, "weight_pde": .2, "iterations": 1000,
            "anchors": len(anchor_indices), "collocation": 18000,
            "lr": None, "steps": None,
        },
    })
    write(run / "manifest.json", {"anchor_indices": anchor_indices})
    write(run / "selection.json", {"iteration": 0, "score": .125})
    write(run / "history.json", [
        {"iteration": 0, "objective": .003, "validation_iv_rmse": .001,
         "validation_parameter_rmse": .125},
        {"iteration": 1000, "objective": .001, "validation_iv_rmse": .0005,
         "validation_parameter_rmse": .2},
    ])
    write(run / "complete.json", {"iterations": 1000, "seconds": 42.})
    truth = [1., .05, .2, -.5, .04, 4., .1, .4, -.3, .06]
    estimate = [value - .0625 if j % 5 == 3 else value * 1.125
                for j, value in enumerate(truth)]
    write(run / "recovery_validation_000000.json", [
        {"case": i, "true": truth, "estimated": estimate, "success": True}
        for i in range(4)
    ])
    # Deliberately distinct file bytes detect a fallback to latest/model weights.
    selected_bytes = b"test fixture: selected iteration-zero weights"
    (run / "iteration_000000.safetensors").write_bytes(selected_bytes)
    (run / "iteration_001000.safetensors").write_bytes(b"test fixture: later weights")
    (run / "model.safetensors").write_bytes(b"test fixture: do not infer a fallback hash")

    result = collect_run(run)
    summary = result["summary"]
    assert summary["update_kind"] == "L-BFGS iteration"
    assert summary["selected_step"] == 0
    assert summary["latest_recorded_step"] == 1000
    assert [row["step"] for row in result["history"]] == [0, 1000]
    assert summary["selected_training_loss"] == .003
    assert summary["selected_validation_iv_rmse"] == .001
    assert summary["selected_validation_parameter_rmse"] == .125
    assert summary["recomputed_validation_parameter_rmse"] == pytest.approx(.125)
    assert summary["selected_checkpoint_path"] == str(run / "iteration_000000.safetensors")
    assert summary["selected_checkpoint_sha256"] == hashlib.sha256(selected_bytes).hexdigest()
    assert summary["weight_decay"] == 0.
    assert result["config"]["resample_every"] == 0
    assert summary["requested_steps"] == 1000
    assert summary["fixed_training_subset"] == len(anchor_indices) == 16384
    assert summary["training_usable"] == 131072
    assert summary["collocation_pool"] == 18000
    assert len(result["selected_parameters"]) == 40
    assert not any("missing" in warning or "no matching history" in warning
                   for warning in summary["warnings"])
