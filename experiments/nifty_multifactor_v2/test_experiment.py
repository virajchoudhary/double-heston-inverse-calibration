import json
from pathlib import Path
import sys
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from literature_exact import Grid,admissible,adaptive,exact,black,fit_bs,bs_predict
from src.mentor_dh_pinn.torch_pricer import price_call,price_call_single
from src.double_heston import price_double_heston_call

C=json.loads((Path(__file__).parent/'config.json').read_text())
P=np.array(C['published_double_slow_first']); S=np.array(C['published_single'])


def test_shock_convention_and_project_analogue():
    r=P[[3,8]]
    assert (r*r).sum()>1
    mat=np.eye(4);mat[0,1]=mat[1,0]=r[0];mat[2,3]=mat[3,2]=r[1]
    assert np.linalg.eigvalsh(mat).min()>0
    q=P.copy();q[[3,8]]=r*.95/np.linalg.norm(r)
    admissible(q)
    for x,t in [(0.,.1),(-.2,.5),(.3,2.)]:
        a=exact(q,np.array([x]),np.array([t]))[0]
        b=price_double_heston_call(1.,np.exp(-x),t,0.,0.,q,node_count=128)
        np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)


@pytest.mark.parametrize('p',[P,S])
def test_prices_bounds_parity_monotonicity_and_alternative_engines(p):
    for s in [.7686,1.,3.0746]:
        q=p.copy().reshape(-1,5);q[:,[1,4]]*=s;q[:,2]*=np.sqrt(s);q=q.ravel()
        admissible(q)
        for t in [7/365,30/365,.5,2.]:
            K=np.linspace(.7,1.3,31);x=-np.log(K);tt=np.full(len(K),t)
            a=exact(q,x,tt)
            assert np.isfinite(a).all() and (a>=np.maximum(1-K,0)-1e-9).all() and (a<=1+1e-9).all()
            assert (np.diff(a)<=1e-9).all()
            puts=a-1+K
            np.testing.assert_allclose(a-puts,1-K,atol=1e-14)
            fn=price_call if len(q)==10 else price_call_single
            b=fn(torch.tensor(q),torch.ones(len(K),dtype=torch.float64),torch.tensor(K),torch.tensor(tt),torch.zeros(len(K),dtype=torch.float64),torch.zeros(len(K),dtype=torch.float64),node_count=128).numpy()
            np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
        for x,t in [(-.26,7/365),(0.,.1),(.35,2.)]:
            a=exact(q,np.array([x]),np.array([t]))[0]
            b,_=adaptive(q,x,t,1e-10);c,_=adaptive(q,x,t,1e-11)
            np.testing.assert_allclose([a,c],b,atol=1e-8,rtol=0)


def test_removed_factor_reduces_to_single_and_terminal_limit():
    q=np.r_[.1,1e-10,1e-6,0.,1e-10,S]
    x=np.array([-.1,0.,.1]);t=np.array([.1,.2,2.])
    np.testing.assert_allclose(exact(q,x,t),exact(S,x,t),atol=1e-8,rtol=0)
    for p in [P,S]:
        atzero=Grid(x,np.zeros(3))(p)
        np.testing.assert_array_equal(atzero,np.maximum(1-np.exp(-x),0))
        small=np.array([adaptive(p,float(xx),1e-7)[0] for xx in x])
        assert np.max(abs(small-atzero))<.0001


def test_bs_fits_do_not_need_heldout_targets():
    x=np.linspace(-.2,.2,80);t=np.repeat([.1,.3,.6,1.],20)
    y=black(x,t,.2)
    for term in [False,True]:
        fitted=fit_bs(x,t,y,term)
        np.testing.assert_allclose(bs_predict(fitted,x,t),y,atol=1e-8,rtol=0)
