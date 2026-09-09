"""One deeper conditional pricing PINN with function-preserving residual layers.

The original five hidden layers remain, followed by trainable two-layer residual
blocks. Zero-initialized block outputs preserve a pretrained price function at
initialization. This is one network, not an ensemble or an exact-pricer hybrid.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import torch

from .regular_pinn import RegularVariancePINN
from .regular_pinn_torch import TorchRegularVariancePINN


class ResidualBlock(nn.Module):
    def __init__(self, width, inner_width):
        super().__init__()
        self.up = nn.Linear(width, inner_width)
        self.down = nn.Linear(inner_width, width)
        self.down.weight = mx.zeros_like(self.down.weight)
        self.down.bias = mx.zeros_like(self.down.bias)

    def __call__(self, x):
        return x + .1 * self.down(mx.tanh(self.up(x)))


class DeepRegularVariancePINN(RegularVariancePINN):
    def __init__(self, factors=2, width=160, depth=5, *, residual_blocks=3,
                 residual_width=384, **kwargs):
        if residual_blocks < 1 or residual_width < 1:
            raise ValueError('Positive residual block count and width required')
        super().__init__(factors, width, depth, **kwargs)
        self.residual_width = residual_width
        self.residual_blocks = [ResidualBlock(width, residual_width) for _ in range(residual_blocks)]
        self.hidden_layer_count = depth + 2 * residual_blocks

    def correction(self, coords, structural):
        h = self.features(coords, structural)
        for layer in self.hidden:
            h = mx.tanh(layer(h))
        for block in self.residual_blocks:
            h = block(h)
        return self.correction_limit * mx.tanh(self.head(h)[..., 0])


class TorchResidualBlock(torch.nn.Module):
    def __init__(self, width, inner_width):
        super().__init__()
        self.up = torch.nn.Linear(width, inner_width)
        self.down = torch.nn.Linear(inner_width, width)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)

    def forward(self, x):
        return x + .1 * self.down(torch.tanh(self.up(x)))


class TorchDeepRegularVariancePINN(TorchRegularVariancePINN):
    def __init__(self, factors=2, width=160, depth=5, *, residual_blocks=3,
                 residual_width=384, **kwargs):
        super().__init__(factors, width, depth, **kwargs)
        self.residual_width = residual_width
        self.residual_blocks = torch.nn.ModuleList([TorchResidualBlock(width, residual_width) for _ in range(residual_blocks)])
        self.hidden_layer_count = depth + 2 * residual_blocks
        self.double()

    def correction(self, coords, structural):
        h = self.features(coords, structural)
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        for block in self.residual_blocks:
            h = block(h)
        return self.correction_limit * torch.tanh(self.head(h)[..., 0])

    @classmethod
    def from_mlx(cls, model):
        keys = ('factors', 'width', 'depth', 'tau_min', 'tau_max', 'x_half_width', 'correction_limit')
        net = cls(**{k: getattr(model, k) for k in keys}, residual_blocks=len(model.residual_blocks),
                  residual_width=model.residual_width)
        state = {key: torch.tensor(np.asarray(value).copy(), dtype=torch.float64)
                 for key, value in tree_flatten(model.parameters())}
        net.load_state_dict(state, strict=True)
        net.eval()
        net.requires_grad_(False)
        return net
