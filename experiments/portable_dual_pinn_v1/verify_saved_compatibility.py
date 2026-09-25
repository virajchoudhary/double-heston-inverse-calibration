"""Prove the shared-input fix does not change any saved batched predictions."""
import importlib.util
import json
import numpy as np
import torch
from run_pilot import HERE,structural,T,PortableVariancePINN,MaturityDualPINN


def main():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('src.mentor_dh_pinn._training_snapshot',HERE/'model_training_snapshot.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    for folder in ['pilot','extended_stratified']:
        a=np.load(HERE/folder/'development.npz');z=T(a['coords']);st=structural(a['params'])
        for kind,cls,priorcls in [('single_branch',PortableVariancePINN,old.PortableVariancePINN),
                                  ('dual_branch',MaturityDualPINN,old.MaturityDualPINN)]:
            for seed in [17,43]:
                state=torch.load(HERE/folder/f'{kind}_{seed}.pt',weights_only=True)
                net,prior=cls(),priorcls();net.load_state_dict(state);prior.load_state_dict(state)
                with torch.no_grad():
                    new=net.price(z,st)*torch.exp(-z[:,0]);before=prior.price(z,st)*torch.exp(-z[:,0])
                torch.testing.assert_close(new,before,atol=0,rtol=0)
                saved=np.load(HERE/folder/f'{kind}_{seed}_development_predictions.npz')['pred']
                np.testing.assert_array_equal(new.numpy(),saved)
                rows.append({'run':folder,'model':kind,'seed':seed,'prices_compared':len(saved),'max_change':0.0})
    (HERE/'saved_compatibility.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('All 8 saved model/seed evaluations are bitwise unchanged:',sum(r['prices_compared'] for r in rows),'prices')


if __name__=='__main__':main()
