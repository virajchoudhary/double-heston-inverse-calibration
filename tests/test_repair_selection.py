"""Holdout prices cannot influence encoder inputs through the validation adapter."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

path=Path(__file__).resolve().parents[1]/"scripts/mentor_dh_pinn/run_finetune.py"
spec=importlib.util.spec_from_file_location("repair_driver",path)
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_validation_adapter_removes_all_heldout_quote_values():
    d={k:np.ones((1,6)) for k in ("spot","strike","tau","rate","carry","price","vega","quote_sigma","mask")}
    d.update(n_quotes=np.array([6]),iv_noise=np.array([.01]),holdout_mask=np.array([[0,0,1,0,0,1]]))
    before,target=driver.validation_batches(d,np.array([0]))
    assert target["mask"].sum()==2 and before["mask"].sum()==4
    for k in ("spot","strike","tau","rate","carry","price","vega","quote_sigma"):
        d[k][0,[2,5]]=1e30
    after,_=driver.validation_batches(d,np.array([0]))
    for key in before:
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
