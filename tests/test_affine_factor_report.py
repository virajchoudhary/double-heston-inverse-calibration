import hashlib
import json

import numpy as np
import pytest
import torch

from scripts.mentor_dh_pinn.report_affine_factor_pinn import audit_training


@pytest.mark.parametrize('metric',['validation_coefficient_rmse','validation_vega_price_rmse'])
def test_training_audit_rejects_relabelled_overlapping_splits(tmp_path,metric):
    """A correct file hash does not excuse train/validation overlap."""
    config={'data':str(tmp_path),'selection_metric':metric}
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'complete.json').write_text('{}')
    history=[{'step':0,'validation_coefficient_rmse':.1,'validation_vega_price_rmse':.2},
             {'step':1,'validation_coefficient_rmse':.2,'validation_vega_price_rmse':.1}]
    selection=min(history,key=lambda r:r[metric])
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    (tmp_path/'history.json').write_text(json.dumps(history))
    state={'weight':torch.tensor([1.],dtype=torch.float64)}
    torch.save(state,tmp_path/f'step_{selection["step"]:06d}.pt')
    torch.save(state,tmp_path/'model.pt')
    (tmp_path/'source_snapshot.json').write_text(json.dumps({'train.py':'# frozen training source\n'}))
    source_hashes={'train.py':hashlib.sha256(b'# frozen training source\n').hexdigest()}
    hashes={}
    for i,split in enumerate(('train','validation','collocation')):
        q=np.array([[.1+i*.1,1.,.2,-.4,1.,0.],[.15+i*.1,2.,.3,-.3,2.,1.]])
        np.savez(tmp_path/f'{split}.npz',q=q,targets=np.zeros((2,4)))
        hashes[split]=hashlib.sha256((tmp_path/f'{split}.npz').read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps({'input_sha256':hashes,'source_sha256':source_hashes}))
    manifest={'checkpoint':str(tmp_path/'model.pt'),'training_config':config}
    assert audit_training(manifest)=={'train':2,'validation':2,'collocation':2}
    (tmp_path/'selection.json').write_text(json.dumps(max(history,key=lambda r:r[metric])))
    with pytest.raises(AssertionError):audit_training(manifest)
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    torch.save({'weight':state['weight']+1},tmp_path/'model.pt')
    with pytest.raises(AssertionError,match='Assessed weights differ'):audit_training(manifest)
    torch.save(state,tmp_path/'model.pt')
    (tmp_path/'source_snapshot.json').write_text(json.dumps({'train.py':'# changed training source\n'}))
    with pytest.raises(AssertionError,match='Training source snapshot changed'):audit_training(manifest)
    (tmp_path/'source_snapshot.json').write_text(json.dumps({'train.py':'# frozen training source\n'}))
    (tmp_path/'validation.npz').write_bytes((tmp_path/'train.npz').read_bytes())
    hashes['validation']=hashes['train']
    (tmp_path/'manifest.json').write_text(json.dumps({'input_sha256':hashes,'source_sha256':source_hashes}))
    with pytest.raises(AssertionError):audit_training(manifest)
