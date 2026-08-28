"""Mentor-aligned Double Heston forward PINN baseline (V1).

This package is deliberately separate from the frozen ``src.model3_pde``
research scaffold.  It trains a conditional European-call forward map with
explicit data, PDE, terminal, and boundary losses.
"""

from .config import BaselineConfig, load_baseline_config
from .model import DoubleHestonForwardPINN
from .parameter_source import ParameterSource, select_first_eligible_train_record

__all__ = [
    "BaselineConfig",
    "DoubleHestonForwardPINN",
    "ParameterSource",
    "load_baseline_config",
    "select_first_eligible_train_record",
]
