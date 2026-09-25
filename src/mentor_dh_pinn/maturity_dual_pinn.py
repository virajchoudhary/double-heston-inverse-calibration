"""Asset-independent European pricing PINN; experimental, not a market guarantee.

Reuses the audited Black output map and Double Heston AD residual. Unlike the
legacy inverse DualPINN this is a forward PDE surrogate with arbitrary quote
geometry, not a fixed-grid ten-parameter estimator. Structural inputs are fixed
when differentiating the PDE. No quote prices enter the feature map.
"""
import math
import torch
from torch import nn
from .regular_pinn_torch import TorchRegularVariancePINN, expected_variance


class PortableVariancePINN(TorchRegularVariancePINN):
    """Single-branch control using scale-free features shared by both candidates."""

    def __init__(self, width=96, depth=3, tau_min=3/365, tau_max=2.,
                 x_half_width=1., correction_limit=1.8):
        super().__init__(2, width, depth, tau_min, tau_max, x_half_width, correction_limit)

    def features(self, coords, structural):
        x, tau = coords[..., 0], coords[..., -1]
        vb = expected_variance(coords, structural)
        z = x / torch.sqrt(tau * vb + x*x/64 + 1e-12)
        zt = torch.tanh(z / 1.5)
        f = [x / self.x_half_width, z/8, zt,
             2*torch.log(tau/self.tau_min)/math.log(self.tau_max/self.tau_min)-1]
        skew = torch.zeros_like(x)
        # Dimensionless features avoid imposing equity-specific variance scales.
        for i in range(2):
            v = coords[..., i+1]
            k, theta, sigma, rho = structural[..., i, :].unbind(-1)
            nu = torch.tanh(sigma*torch.sqrt(tau)/torch.sqrt(vb+1e-12))
            eta = sigma/torch.sqrt(2*k*theta)
            f += [torch.log(v/vb), torch.log1p(k*tau), torch.log(theta/vb),
                  rho+torch.zeros_like(x), torch.tanh(k*tau), nu, rho*nu,
                  torch.tanh(.5*torch.log(v/theta)), torch.log1p(eta)+torch.zeros_like(x)]
            skew = skew + rho*nu
        f += [2*coords[..., 1]/coords[..., 1:3].sum(-1)-1, zt*skew]
        return torch.stack(f, -1)


class MaturityDualPINN(PortableVariancePINN):
    """Smooth short/long experts; their joint output obeys the same PDE loss.

    The fixed smooth gate covers both experts continuously; no hard expiry split
    or dropout that could destabilise second derivatives. More expressive does
    not mean empirically better: promote only after a declared comparison.
    """

    def __init__(self, width=64, depth=3, transition_years=90/365, **kwargs):
        if transition_years <= 0:
            raise ValueError('transition_years must be positive')
        super().__init__(width=width, depth=depth, **kwargs)
        self.transition_years = transition_years
        self.long_hidden = nn.ModuleList(nn.Linear(a.in_features, a.out_features)
                                         for a in self.hidden)
        self.long_head = nn.Linear(width, 1)
        self.double()

    def correction(self, coords, structural):
        a = b = self.features(coords, structural)
        for layer in self.hidden:
            a = torch.tanh(layer(a))
        for layer in self.long_hidden:
            b = torch.tanh(layer(b))
        gate = torch.sigmoid(2*torch.log(coords[..., -1]/self.transition_years))
        h = (1-gate)*self.head(a)[..., 0] + gate*self.long_head(b)[..., 0]
        return self.correction_limit*torch.tanh(h)


def european_price(model, *, forward, strike, tau, discount, params,
                   option_type='call', exercise='european'):
    """Return price in the supplied forward/strike currency, not coin units.

    params: slow-first [kappa,theta,sigma,rho,v0] repeated twice. Forward and
    discount MUST come from independent market/carry data, never held-out option
    targets. For coin-denominated data the caller must explicitly convert using
    the venue's documented settlement convention. No ticker or grid is assumed.
    Negative rates (discount > 1) are allowed. No Feller/joint-rho disk restriction.
    """
    if exercise != 'european' or option_type not in ('call', 'put'):
        raise ValueError('Only European calls and puts are supported')
    anchor = next(model.parameters())
    cast = lambda a: torch.as_tensor(a, dtype=anchor.dtype, device=anchor.device)
    p = cast(params)
    if p.ndim < 1 or p.shape[-1] != 10:
        raise ValueError('Expected ten slow-first Double Heston parameters')
    F, K, t, D, *columns = torch.broadcast_tensors(
        cast(forward), cast(strike), cast(tau), cast(discount), *p.unbind(-1))
    p = torch.stack(columns, -1)
    if not all(torch.isfinite(a).all() for a in [F,K,t,D,p]):
        raise ValueError('Nonfinite inputs')
    if ((F<=0)|(K<=0)|(t<0)|(D<=0)).any():
        raise ValueError('Positive forward, strike, discount and nonnegative tau required')
    factors = p.reshape(*p.shape[:-1], 2, 5)
    if (factors[..., [0,1,2,4]]<=0).any() or (factors[...,3].abs()>=1).any():
        raise ValueError('Invalid variance-model parameters')
    if (p[...,0]>=p[...,5]).any():
        raise ValueError('Require kappa_slow < kappa_fast')
    x = torch.log(F/K)
    if ((t>0)&((t<model.tau_min)|(t>model.tau_max))).any() or (x.abs()>model.x_half_width).any():
        raise ValueError('Outside declared maturity/moneyness domain; do not silently extrapolate')
    coords = torch.stack([x,p[...,4],p[...,9],t], -1)
    call = D*K*model.price(coords, factors[...,:4])
    return call if option_type == 'call' else call-D*(F-K)
