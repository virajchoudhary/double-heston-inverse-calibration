"""Float64 inference copy of the regular MLX PINN, with the same learned weights.

Training runs on Apple Silicon. Calibration uses this identical network in
float64 so SciPy sees smooth values and gradients below float32 resolution.
No characteristic function, exact price, or parameter-polish stage is involved.
"""
import math
import numpy as np
import torch
from torch import nn


def mean_reversion_ratio(u):
    return -torch.expm1(-u)/u


def expected_variance(coords,structural):
    k,theta=structural[...,0],structural[...,1]
    v=coords[...,1:-1];tau=coords[...,-1:]
    return (theta+(v-theta)*mean_reversion_ratio(k*tau)).sum(-1)


def unit_log(x,low,high):return 2*(torch.log(x)-math.log(low))/math.log(high/low)-1


def black_call(x, total_variance):
    """Normalized Black call with analytic payoff at zero total variance."""
    positive = total_variance > 0
    safe = torch.where(positive, total_variance, torch.ones_like(total_variance))
    root = torch.sqrt(safe)
    d1 = x / root + 0.5 * root
    d2 = d1 - root
    cdf = lambda value: 0.5 * torch.erfc(-value / math.sqrt(2.0))
    price = torch.exp(x) * cdf(d1) - cdf(d2)
    return torch.where(positive, price, torch.clamp_min(torch.expm1(x), 0.0))


class TorchRegularVariancePINN(nn.Module):
    def __init__(self,factors=2,width=160,depth=5,tau_min=7/365,tau_max=2.,
                 x_half_width=3.5,correction_limit=1.8):
        super().__init__();self.factors=factors;self.width=width;self.depth=depth
        self.tau_min=tau_min;self.tau_max=tau_max;self.x_half_width=x_half_width
        self.correction_limit=correction_limit
        sizes=[13 if factors==1 else 24]+[width]*depth
        self.hidden=nn.ModuleList(nn.Linear(a,b) for a,b in zip(sizes[:-1],sizes[1:]))
        self.head=nn.Linear(width,1);self.double()

    @classmethod
    def from_mlx(cls,model):
        if hasattr(model, 'residual_blocks'):
            from .deep_regular_pinn import TorchDeepRegularVariancePINN
            return TorchDeepRegularVariancePINN.from_mlx(model)
        net=cls(**{k:getattr(model,k) for k in ("factors","width","depth","tau_min","tau_max",
                                                 "x_half_width","correction_limit")})
        with torch.no_grad():
            for source,target in zip([*model.hidden,model.head],[*net.hidden,net.head]):
                target.weight.copy_(torch.tensor(np.asarray(source.weight).copy(),dtype=torch.float64))
                target.bias.copy_(torch.tensor(np.asarray(source.bias).copy(),dtype=torch.float64))
        net.eval();net.requires_grad_(False)
        return net

    def features(self,coords,structural):
        x,tau=coords[...,0],coords[...,-1];vb=expected_variance(coords,structural)
        z=x/torch.sqrt(tau*vb+x*x/64+1e-12);zt=torch.tanh(z/1.5)
        features=[x/self.x_half_width,z/8,zt,unit_log(tau,self.tau_min,self.tau_max)]
        skew=torch.zeros_like(x)
        for i in range(self.factors):
            v=coords[...,i+1];k,t,s,r=[structural[...,i,j]+torch.zeros_like(x) for j in range(4)]
            nu=torch.tanh(s*torch.sqrt(tau)/torch.sqrt(vb+1e-9));eta=s/torch.sqrt(2*k*t)
            features += [unit_log(v,.01,.3),unit_log(k,.15,12),unit_log(t,.01,.3),r,
                    torch.tanh(k*tau),nu,r*nu,torch.tanh(.5*torch.log(v/t)),2*torch.log1p(eta)/math.log(4)-1]
            skew=skew+r*nu
        if self.factors==2:features += [2*coords[...,1]/coords[...,1:-1].sum(-1)-1,zt*skew]
        return torch.stack(features,dim=-1)

    def correction(self,coords,structural):
        h=self.features(coords,structural)
        for layer in self.hidden:h=torch.tanh(layer(h))
        return self.correction_limit*torch.tanh(self.head(h)[...,0])

    def iv(self,coords,structural):
        return torch.sqrt(expected_variance(coords,structural))*torch.exp(self.correction(coords,structural))

    def forward(self,coords,structural):return self.iv(coords,structural)

    def log_total_variance(self, coords, structural):
        """Log implied total variance for positive tau and variance states."""
        return (torch.log(coords[..., -1]) + torch.log(expected_variance(coords, structural))
                + 2.0 * self.correction(coords, structural))

    def price(self, coords, structural):
        """Normalized forward call c=C/(K exp(-r*tau)), exact payoff at tau=0."""
        tau = coords[..., -1]
        safe_tau = torch.where(tau > 0, tau, torch.ones_like(tau))
        safe_coords = torch.cat([coords[..., :-1], safe_tau[..., None]], dim=-1)
        w = torch.exp(self.log_total_variance(safe_coords, structural))
        return black_call(coords[..., 0], torch.where(tau > 0, w, torch.zeros_like(w)))


@torch.enable_grad()
def residual(model, coords, structural):
    """Float64-compatible regular Heston PDE residual with differentiable weights.

    The residual equals ``(c_tau - generator(c)) / (Black_c_w * scale)``;
    ``scale=sum(v_i)+expected_average_variance``. Factor variances are independent,
    so the only mixed derivatives are x-v_i, with coefficients rho_i*sigma_i*v_i.
    All network-weight gradients are retained for PDE training. If coordinate
    gradients are not enabled, a local coordinate leaf is created without
    changing the caller's tensor. Structural parameters stay fixed with respect
    to the coordinate derivatives.
    """
    if not coords.requires_grad:
        coords = coords.detach().requires_grad_(True)

    def derivative(value):
        return torch.autograd.grad(value.sum(), coords, create_graph=True)[0]

    ell = model.log_total_variance(coords, structural)
    first = derivative(ell)
    second_x = derivative(first[..., 0])
    w, x = torch.exp(ell), coords[..., 0]
    lx, lt, lxx = first[..., 0], first[..., -1], second_x[..., 0]
    a = 0.5 * x * x - 0.125 * w * w - 0.5 * w
    diffusion_x = (2.0 + 2.0 * (0.5 * w - x) * lx + a * lx * lx
                   + w * (lxx + lx * lx) - w * lx)
    total = coords[..., 1:-1].sum(-1)
    value = w * lt - 0.5 * total * diffusion_x
    for i in range(model.factors):
        j = i + 1
        lv, lxv = first[..., j], second_x[..., j]
        lvv = derivative(first[..., j])[..., j]
        v = coords[..., j]
        kappa, theta, sigma, rho = [structural[..., i, k] for k in range(4)]
        cross = (0.5 * w - x) * lv + a * lx * lv + w * (lxv + lx * lv)
        diffusion_v = a * lv * lv + w * (lvv + lv * lv)
        value = (value - rho * sigma * v * cross - 0.5 * sigma * sigma * v * diffusion_v
                 - kappa * (theta - v) * w * lv)
    scale = total + expected_variance(coords, structural)
    convexity = ((1.0 - 0.5 * x * lx)**2 - 0.25 * w * lx * lx
                 - w * w * lx * lx / 16.0 + 0.5 * w * (lxx + lx * lx))
    return value / scale, {"ell": ell, "w": w, "l_x": lx, "l_tau": lt,
                           "l_xx": lxx, "convexity": convexity, "scale": scale}
