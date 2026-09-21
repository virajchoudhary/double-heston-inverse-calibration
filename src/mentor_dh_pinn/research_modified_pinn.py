"""Checkpoint-preserving correction with the published Wang modified MLP.

The core recurrence follows GradientPathologiesPINNs M3/M4 and JaxPI
ModifiedMlp: H <- tanh(W H+b)*U + (1-tanh(W H+b))*V.
Using this core as an incremental correction is our adaptation, not a claim
that the paper used pretrained option-price residual corrections.
"""
import torch
from torch import nn
from .regular_pinn_torch import TorchRegularVariancePINN


class ModifiedMLP(nn.Module):
    def __init__(self, inputs=26, width=96, depth=5, modified=True, adaptive=False):
        super().__init__()
        self.modified=modified
        self.layers=nn.ModuleList([nn.Linear(inputs,width)]+[nn.Linear(width,width) for _ in range(depth-1)])
        if modified:
            self.u=nn.Linear(inputs,width);self.v=nn.Linear(inputs,width)
        self.scales=nn.Parameter(torch.ones(depth)) if adaptive else None
        self.head=nn.Linear(width,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.double()

    def forward(self, features):
        h=features
        if self.modified:u,v=torch.tanh(self.u(features)),torch.tanh(self.v(features))
        for i,layer in enumerate(self.layers):
            h=torch.tanh(layer(h)*(self.scales[i] if self.scales is not None else 1.))
            if self.modified:h=h*u+(1-h)*v
        return self.head(h)[...,0]


class ResearchCorrectionPINN(TorchRegularVariancePINN):
    def __init__(self,base,center,scale,modified=True,adaptive=False):
        nn.Module.__init__(self)
        self.base=base.requires_grad_(False)
        for key in ('factors','width','depth','tau_min','tau_max','x_half_width','correction_limit'):
            setattr(self,key,getattr(base,key))
        self.register_buffer('center',torch.as_tensor(center,dtype=torch.float64))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float64))
        if not torch.isfinite(self.scale).all() or not (self.scale>0).all():raise ValueError('Invalid feature scale')
        self.core=ModifiedMLP(width=96 if modified else 104,modified=modified,adaptive=adaptive)

    def branch_features(self,z,s):
        f=self.base.base.features(z,s)
        decay=torch.exp(-s[..., :,0]*z[...,-1:])
        return (torch.cat([f,decay],-1)-self.center)/self.scale

    def correction(self,z,s):
        return self.base.correction(z,s)+.05*torch.tanh(self.core(self.branch_features(z,s)))
