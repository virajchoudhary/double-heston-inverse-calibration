import numpy as np
import pytest
import torch
from src.mentor_dh_pinn.identifiability import canonicalize_parameters,match_factors,recovery_metrics,SWAP
from src.mentor_dh_pinn.regular_pinn_data import decode_unit,teacher_labels
from src.mentor_dh_pinn.recovery_diagnostics import price_and_jacobian,physical_to_unit,information_diagnostics


def test_factor_canonicalization_and_permutation_gate():
    units=np.random.default_rng(91201).uniform(.01,.99,(30,10));p=decode_unit(units,2)
    np.testing.assert_array_equal(canonicalize_parameters(p),p)
    np.testing.assert_array_equal(canonicalize_parameters(p[:,SWAP]),p)
    assert not recovery_metrics(p[:,SWAP],p,permutation_invariant=False)['all_ten'].any()
    assert recovery_metrics(p[:,SWAP],p)['all_ten'].all()
    tie=p[0].copy();tie[5]=tie[0]
    np.testing.assert_array_equal(canonicalize_parameters(tie),canonicalize_parameters(tie[SWAP]))
    with pytest.raises(ValueError):canonicalize_parameters(np.zeros(9))


def test_exact_factor_symmetry_jacobian_and_inverse_map():
    torch.set_num_threads(1)
    unit=np.array([.4,.6,.5,.3,.7,.6,.3,.4,.2,.8]);p=decode_unit(unit,2)
    np.testing.assert_allclose(physical_to_unit(p),unit,atol=1e-14)
    x=np.array([-.18,0,.17]);t=np.array([.08,.5,2.])
    price,j=price_and_jacobian(p,x,t);swapped,sj=price_and_jacobian(p[SWAP],x,t)
    np.testing.assert_allclose(price,swapped,atol=1e-14,rtol=0)
    np.testing.assert_allclose(j,sj[:,SWAP],atol=1e-13,rtol=1e-11)
    for k in range(10):
        h=1e-5*max(abs(p[k]),.01);delta=np.eye(10)[k]*h
        fd=(price_and_jacobian(p+delta,x,t)[0]-price_and_jacobian(p-delta,x,t)[0])/(2*h)
        np.testing.assert_allclose(j[:,k],fd,atol=2e-8,rtol=2e-5)


def test_information_scaling_and_known_rank():
    p=decode_unit(np.full(10,.5),2);j=np.eye(10)
    info=information_diagnostics(j,p,standard_deviation=np.ones(10)*.01)
    tol=.05*abs(p);tol[3::5]=.05
    np.testing.assert_allclose(info['gram'],np.diag((tol/.01)**2))
    assert info['numerical_rank']==10
    j[:,-1]=j[:,0]
    info=information_diagnostics(j,p)
    assert info['numerical_rank']==9
    assert info['conditional_sensitivity'][0]<1e-12
