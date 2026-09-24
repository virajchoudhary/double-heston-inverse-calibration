"""Architecture variants for the v5 ablation.

Every variant keeps the v4 design that works: the 24 engineered features, the analytic
expected-variance baseline, the bounded correction and the Black pricing layer. Only the
learned body between features and head changes, so the frozen PDE residual in
regular_pinn_torch.residual() applies unchanged to all of them.
"""
import math
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


class AdaptiveTanh(nn.Module):
    """tanh(a * z) with one learned scalar per block (Jagtap, Kawaguchi and Karniadakis)."""

    def __init__(self, adaptive):
        super().__init__(); self.a = nn.Parameter(torch.ones((), dtype=torch.float64), requires_grad=bool(adaptive))

    def forward(self, z): return torch.tanh(self.a * z)


class ResidualBody(nn.Module):
    """Projection, then h <- h + alpha_l * act(Linear(h)); alpha_l learned, initialised small."""

    def __init__(self, n_in, width, depth, adaptive):
        super().__init__()
        self.inp = nn.Linear(n_in, width); self.act0 = AdaptiveTanh(adaptive)
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.acts = nn.ModuleList(AdaptiveTanh(adaptive) for _ in range(depth - 1))
        self.alpha = nn.Parameter(torch.full((depth - 1,), .1, dtype=torch.float64))

    def forward(self, f):
        h = self.act0(self.inp(f))
        for j, (lin, act) in enumerate(zip(self.layers, self.acts)): h = h + self.alpha[j] * act(lin(h))
        return h


class GatedBody(nn.Module):
    """Improved fully-connected architecture of Wang, Teng and Perdikaris (gradient-flow paper):
    U = act(X W1 + b1), V = act(X W2 + b2); H1 = act(X W + b);
    Z = act(H_k W_k + b_k); H_{k+1} = (1 - Z) * U + Z * V."""

    def __init__(self, n_in, width, depth, adaptive):
        super().__init__()
        self.u = nn.Linear(n_in, width); self.v = nn.Linear(n_in, width); self.inp = nn.Linear(n_in, width)
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.acts = nn.ModuleList(AdaptiveTanh(adaptive) for _ in range(depth + 2))

    def forward(self, f):
        U = self.acts[0](self.u(f)); V = self.acts[1](self.v(f)); h = self.acts[2](self.inp(f))
        for j, lin in enumerate(self.layers):
            z = self.acts[3 + j](lin(h)); h = (1 - z) * U + z * V
        return h


class PlainBody(nn.Module):
    """The locked v4 body: Linear(24,W) then depth-1 further Linear(W,W), tanh after each."""

    def __init__(self, n_in, width, depth, adaptive):
        super().__init__()
        sizes = [n_in] + [width] * depth
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.acts = nn.ModuleList(AdaptiveTanh(adaptive) for _ in range(depth))

    def forward(self, f):
        h = f
        for lin, act in zip(self.layers, self.acts): h = act(lin(h))
        return h


BODIES = {'plain': PlainBody, 'residual': ResidualBody, 'gated': GatedBody}


class V5Pinn(TorchRegularVariancePINN):
    """Same inputs, features, baseline, bounded correction and Black layer as v4; swappable body."""

    def __init__(self, body='plain', adaptive=False, factors=2, width=256, depth=5,
                 tau_min=7 / 365, tau_max=2., x_half_width=.36, correction_limit=1.8):
        super().__init__(factors=factors, width=width, depth=depth, tau_min=tau_min, tau_max=tau_max,
                         x_half_width=x_half_width, correction_limit=correction_limit)
        n_in = 13 if factors == 1 else 24
        del self.hidden
        self.body_kind = body; self.adaptive = bool(adaptive)
        self.body = BODIES[body](n_in, width, depth, adaptive)
        self.head = nn.Linear(width, 1)
        self.double()

    def correction(self, coords, structural):
        return self.correction_limit * torch.tanh(self.head(self.body(self.features(coords, structural)))[..., 0])


def count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
