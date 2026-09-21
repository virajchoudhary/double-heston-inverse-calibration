"""Independent inversion agrees at regular prices and resolves tiny positive tails."""
import mpmath as mp
import torch
from scripts.mentor_dh_pinn.evaluate_repair_validation import score_quote, summarize
from scripts.mentor_dh_pinn.diagnose_tail_precision import price_and_iv
from src.mentor_dh_pinn.torch_pricer import price_call


def test_independent_inversion_matches_regular_fourier_price():
    p=[1.,.04,.2,-.5,.04,5.,.02,.3,-.3,.02]
    result=price_and_iv(p,1.,1.,.25,40)
    args=[torch.tensor(x,dtype=torch.float64) for x in (p,1.,1.,.25,0.,0.)]
    expected=float(price_call(*args,node_count=128))
    assert result["status"]=="finite_price_and_iv"
    assert abs(float(result["price"])-expected)<1e-11
    assert abs(mp.mpf(result["tail_interval_contribution"]))<mp.mpf("1e-20")


def test_invalid_prediction_keeps_entire_metric_unavailable():
    bad={"label":"test",**score_quote(-1e-14,.01,1.,1.2,.1,.05)}
    good={"label":"test",**score_quote(.01,.01,1.,1.2,.1,.05)}
    result=summarize([good,bad])
    assert result["quotes"]==2 and result["invalid_iv_quotes"]==1
    assert result["no_arbitrage_violations"]==1
    assert result["iv_rmse_all_quotes"] is None
