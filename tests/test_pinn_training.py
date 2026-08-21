from __future__ import annotations

from pathlib import Path

from models.pinn_model import PhysicsInformedInverseCalibrator
from src.dataset import SurfaceParameterDataset
from src.synthetic_dataset import generate_smoke_test_dataset
from src.train_pinn import train_pinn


def test_pinn_training_smoke(tmp_path: Path) -> None:
    output = tmp_path / "smoke_test"
    frame = generate_smoke_test_dataset(output, n_surfaces=6, seed=42)
    dataset = SurfaceParameterDataset.from_surface_frame(
        frame,
        allow_not_research_data=True,
    )
    model = PhysicsInformedInverseCalibrator(
        input_size=dataset.features.shape[1],
        hidden_sizes=(32, 32),
        dropout=0.0,
    )
    summary = train_pinn(
        model,
        dataset,
        dataset.indices_for_split("train"),
        dataset.indices_for_split("validation"),
        tmp_path / "training_output",
        epochs=1,
        patience=1,
        batch_size=2,
        node_count=16,
    )
    assert summary["best_epoch"] == 1
    assert summary["epochs_completed"] == 1
    assert summary["checkpoint_path"].exists()
