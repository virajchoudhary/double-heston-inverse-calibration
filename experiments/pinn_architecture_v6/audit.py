"""Post-run integrity checks, no training or selection changes."""
import platform
import sys
import numpy as np
import pandas as pd
import scipy
import torch
from .common import *
from .train import verify,ARMS


def main():
    verify();prior.verify();old.verify()
    checked=[]
    for arm in ARMS:
        for seed in SEEDS:
            path=OUT/arm/f'seed_{seed}/weights.pt'
            if not path.exists():continue
            original=torch.load(prior.OUT/f'SHARED_s{seed}/weights.pt',weights_only=True)
            new=torch.load(path,weights_only=True)
            if arm=='A0_CONTINUE':pairs=[(key,key) for key in original if key.startswith('base.')]
            else:pairs=[(key,'base.'+key) for key in original]
            assert all(torch.equal(original[a],new[b]) for a,b in pairs),arm
            assert all(torch.isfinite(v).all() for v in new.values()),arm
            assert sum(v.numel() for k,v in new.items() if k not in ['center','scale'])<=1.25*275684
            checked.append({'arm':arm,'seed':seed,'unchanged_parent_tensors':len(pairs),'checkpoint_sha256':sha(path)})
    pools=[]
    for path in OUT.glob('ARCH_C*/seed_*/rad_*.npz'):
        a=np.load(path);z=a['selected'];prob=a['probability']
        assert len(z)==4096 and len(set(map(tuple,z)))==4096
        assert np.isfinite(z).all() and np.isfinite(prob).all() and (prob>0).all() and abs(prob.sum()-1)<1e-12
        assert (abs(z[:,0])<=.36).all() and (z[:,-1]>=7/365).all() and (z[:,-1]<=2).all()
        assert ((z[:,1]>=.00015)&(z[:,1]<=.18)).all() and ((z[:,2]>=.001)&(z[:,2]<=.22)).all()
        pools.append(str(path.relative_to(ROOT)))
    disjoint=None
    if (OUT/'final_fidelity/data.npz').exists():
        final=np.load(OUT/'final_fidelity/data.npz');keys=set(map(tuple,final['coords']))
        assert len(keys)==4096
        for name in ['train','development','collocation']:
            assert not keys & set(map(tuple,data(name)['coords']))
        for path in OUT.glob('ARCH_C*/seed_*/rad_*.npz'):
            assert not keys & set(map(tuple,np.load(path)['pool']))
        disjoint=True
    if (OUT/'selection.json').exists():
        for path,digest in read(OUT/'selection.json')['files'].items():assert sha(ROOT/path)==digest,path
    save(OUT/'integrity.json',{'utc':old.stamp(),'checks':checked,'verified_RAD_pools':pools,
         'fresh_fidelity_disjoint_from_training_development_collocation_and_RAD':disjoint,
         'parent_v3_v4_v5_sources_preserved':True,'market_quotes_parsed_or_scored':False})
    save(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
         'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'device':'cpu','threads_per_process':1})
    print('Verified parent preservation:',len(checked),'runs; RAD pools:',len(pools),'fresh disjoint:',disjoint)


if __name__=='__main__':main()
