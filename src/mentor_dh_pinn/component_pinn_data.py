"""Broader single-factor training box covering both individual DH factors."""
import math
import numpy as np
import torch
from .regular_pinn_data import coordinates

DOMAIN={'kappa':[.2,12.], 'theta_and_v0':[.006,.30], 'rho':[-.8,.5],
        'feller_ratio':[.15,.90], 'tau':[7/365,2.],
        'purpose':'single-factor building block, not original single-model inverse parameter box'}


def decode_component_unit(u,factors,lib=np):
    if factors!=1:raise ValueError('Component training uses one variance factor')
    k=.2*lib.exp(u[...,0]*math.log(60))
    theta=.006*lib.exp(u[...,1]*math.log(50))
    sigma=(.15+.75*u[...,2])*lib.sqrt(2*k*theta)
    rho=-.8+1.3*u[...,3]
    v=.006*lib.exp(u[...,4]*math.log(50))
    return lib.stack([k,theta,sigma,rho,v],**({'dim':-1} if lib is torch else {'axis':-1}))


def component_coordinates(q,factors,lib=np):
    return coordinates(q,factors,lib,decoder=decode_component_unit)
