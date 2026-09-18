"""Frozen, isolated experiment. Run --help; all old artifacts are read-only."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc
import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'artifacts'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from literature_exact import Grid,exact,admissible,iv,black,fit_bs,bs_predict,fit_sh,batch_teacher
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN,residual
from scripts.mentor_dh_pinn.nifty_fixed_crisis import prepare as clean_market


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def tensor(a):return torch.tensor(np.asarray(a),dtype=torch.float64)


def scaled(p,scale):
    a=np.array(p,float).reshape(-1,5).copy();a[:,[1,4]]*=scale;a[:,2]*=np.sqrt(scale)
    admissible(a.ravel());return a.ravel()


def freeze():
    OUT.mkdir(exist_ok=False)
    c=read(HERE/'config.json');P=np.array(c['published_double_slow_first'])
    bank={'published_double':P.tolist(),'published_single':c['published_single'],'market':{}}
    analog=P.copy();analog[[3,8]]*=c['project_valid_analog_radius']/np.linalg.norm(analog[[3,8]])
    bank['PROJECT_VALID_LITERATURE_ANALOG']={'params':analog.tolist(),'use':'canonical-pricer regression only, not the primary published experiment'}
    for kind,p in [('double',P),('single',c['published_single'])]:
        base=np.array(p)
        bank['market'][kind]=[{'id':f'{kind}_{int(vol*100)}','target_initial_vol':vol,
            'params':scaled(base,vol**2/base.reshape(-1,5)[:,4].sum()).tolist(),
            'label':'PROPOSED_NIFTY_VARIANCE_SCALED_LITERATURE_VARIANT'} for vol in c['market_initial_volatilities']]
    save(OUT/'scenario_bank.json',bank)
    sources=list(HERE.glob('*.py'))+[HERE/'config.json',HERE/'CONTRACT_AUDIT.md']
    sources += [ROOT/p for p in ['src/double_heston.py','src/double_heston_reference.py','src/constraints.py',
        'src/mentor_dh_pinn/regular_pinn_torch.py','src/mentor_dh_pinn/regular_pinn_data.py','src/mentor_dh_pinn/torch_pricer.py',
        'scripts/mentor_dh_pinn/nifty_fixed_crisis.py','scripts/mentor_dh_pinn/audit_crisis_option_coverage.py']]
    snapshot=OUT/'frozen_sources';snapshot.mkdir()
    for p in sources:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    manifest={'frozen_utc':stamp(),'config':c,'scenario_bank_sha256':sha(OUT/'scenario_bank.json'),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'development_exposure':['All dates/results in NIFTY_FIXED_CRISIS_RESULTS_20260917.md','Earlier NIFTY experiments and reserved corpus dates through 2026-08-03'],
        'market_final_opened':False,'convention':'literature_exact','canonical_contract_modified':False}
    save(OUT/'manifest.json',manifest)
    (OUT/'manifest.sha256').write_text(sha(OUT/'manifest.json')+'\n')
    print('FROZEN',sha(OUT/'manifest.json'),flush=True)


def verify():
    a=read(OUT/'manifest.json')
    assert sha(OUT/'manifest.json')==(OUT/'manifest.sha256').read_text().strip()
    assert sha(OUT/'scenario_bank.json')==a['scenario_bank_sha256']
    for path,digest in a['source_hashes'].items():assert sha(ROOT/path)==digest,f'Frozen source changed: {path}'
    return a['config']


def validate():
    verify()
    r=subprocess.run([sys.executable,'-m','pytest','-q',str(HERE/'test_experiment.py'),
        str(ROOT/'tests/test_nifty_fixed_crisis.py'),str(ROOT/'tests/test_crisis_option_coverage.py'),
        str(ROOT/'tests/test_regular_pinn_torch_physics.py')],cwd=ROOT,capture_output=True,text=True)
    (OUT/'exact_tests.txt').write_text(r.stdout+r.stderr)
    save(OUT/'exact_validation.json',{'passed':r.returncode==0,'utc':stamp(),'output_sha256':sha(OUT/'exact_tests.txt')})
    print(r.stdout+r.stderr,flush=True)
    if r.returncode:raise RuntimeError('Phase A failed. No training/evaluation permitted.')


def require_exact():
    c=verify();assert read(OUT/'exact_validation.json')['passed'];return c


def fetch_market(stage):
    c=require_exact()
    if stage=='final':
        assert (OUT/'controlled_results.json').exists() and (OUT/'market_selection.json').exists()
        assert (OUT/'evaluation_lock.json').exists()
    dates=pd.bdate_range(*c['market_'+stage]);dest=OUT/'market_sources';dest.mkdir(exist_ok=True)
    records=[]
    for dt in dates:
        name=f'BhavCopy_NSE_FO_0_0_0_{dt:%Y%m%d}_F_0000.csv.zip'
        url='https://nsearchives.nseindia.com/content/fo/'+name
        path=dest/'raw/nse_fo_bhavcopies'/str(dt.year)/name;path.parent.mkdir(parents=True,exist_ok=True)
        row={'date':f'{dt:%Y-%m-%d}','url':url,'file':str(path.relative_to(dest)),'status':'unavailable'}
        try:
            if path.exists():data=path.read_bytes()
            else:
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req,timeout=20) as response:data=response.read()
                with zipfile.ZipFile(io.BytesIO(data)) as z:assert len(z.namelist())>=1
                path.write_bytes(data)
            row.update(status='downloaded',sha256=sha(path),bytes=len(data),error='')
        except Exception as e:row.update(status=f'http_{e.code}' if isinstance(e,urllib.error.HTTPError) else 'error',error=str(e),sha256='',bytes=0)
        records.append(row);print(stage,row['date'],row['status'],flush=True)
    save(OUT/f'market_{stage}_download.json',{'utc':stamp(),'rows':records})
    # Store both download phases in a source manifest; never synthesize missing quotes.
    combined=[]
    for p in sorted(OUT.glob('market_*_download.json')):combined+=read(p)['rows']
    manifest=dest/'outputs/019fc8a0/nse_source_manifest.csv';manifest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(combined).to_csv(manifest,index=False)
    if sum(r['status']=='downloaded' for r in records)<10:
        save(OUT/f'market_{stage}_status.json',{'status':'BLOCKED_INSUFFICIENT_OFFICIAL_FILES','no_exposed_date_substitution':True});return
    cfg=read(ROOT/'configs/nifty_fixed_crisis.json');cfg['windows']={stage:c['market_'+stage]}
    clean_market(cfg,dest,OUT/f'market_{stage}',HERE/'config.json')
    save(OUT/f'market_{stage}_status.json',{'status':'CLEANED'})


def market_panel(stage):
    p=OUT/f'market_{stage}/clean_quotes.csv'
    if not p.exists():return None
    a=pd.read_csv(p)
    return a[a.date_eligible].copy()


def fit_market_bs(a,term):
    x=a.x.to_numpy();tau=a.tau.to_numpy();y=a.market_call_normalized.to_numpy()*np.exp(-x)
    knots=np.array([7,30,60,100])/365 if term else np.array([30/365])
    def predictions(v):return bs_predict({'knots':knots.tolist(),'volatility':v.tolist()},x,tau)
    wt=a.date.map(a.groupby('date').size()).to_numpy()**-.5
    r=least_squares(lambda v:(predictions(v)-y)*wt/.01,np.full(len(knots),.2),bounds=(.02,2.5),max_nfev=400)
    return {'knots':knots.tolist(),'volatility':r.x.tolist(),'term':term,'success':bool(r.success)}


def market_select():
    c=require_exact();a=market_panel('validation')
    if a is None:print('Market selection blocked: official validation quotes unavailable',flush=True);return
    a=a[a.split.eq('test')].copy()
    if a.date.nunique()<10:raise RuntimeError('Insufficient validation dates; no replacement dates')
    x=a.x.to_numpy();t=a.tau.to_numpy();y=a.market_call_normalized.to_numpy()*np.exp(-x)
    bank=read(OUT/'scenario_bank.json');records=[];chosen={}
    for kind in ['single','double']:
        candidates=[]
        for candidate in bank['market'][kind]:
            pred=exact(candidate['params'],x,t)
            obj=float(pd.Series((pred-y)**2).groupby(a.date.to_numpy()).mean().mean())
            row={**candidate,'validation_equal_date_MSE':obj};records.append(row);candidates.append(row)
        chosen[kind]=min(candidates,key=lambda r:r['validation_equal_date_MSE'])
    # BS family selected on inner chronologically later validation dates only.
    dates=sorted(a.date.unique());cut=dates[int(len(dates)*.7)]
    inner=a[a.date<cut];check=a[a.date>=cut]; bsrows=[]
    for term in [False,True]:
        fitted=fit_market_bs(inner,term)
        yy=check.market_call_normalized.to_numpy()*np.exp(-check.x.to_numpy())
        err=bs_predict(fitted,check.x.to_numpy(),check.tau.to_numpy())-yy
        obj=float(pd.Series(err**2).groupby(check.date.to_numpy()).mean().mean())
        bsrows.append({'term':term,'objective':obj})
    selected=min(bsrows,key=lambda r:r['objective'])['term']
    chosen['bs']=fit_market_bs(a,selected)
    save(OUT/'market_selection.json',{'utc':stamp(),'candidates':records,'selected':chosen,'bs_inner':bsrows,
        'validation_sha256':sha(OUT/'market_validation/clean_quotes.csv'),'manifest_sha256':sha(OUT/'manifest.json')})
    print('Market selection frozen:',chosen,flush=True)


def sample(c,n,seed):
    u=qmc.LatinHypercube(5,seed=seed).random(n); pc=c['pinn']
    x=-.36+.72*u[:,0]
    tau=np.exp(np.log(7/365)+np.log(730/7)*u[:,1])
    vs=np.exp(np.log(pc['slow_state_domain'][0])+np.log(pc['slow_state_domain'][1]/pc['slow_state_domain'][0])*u[:,2])
    vf=np.exp(np.log(pc['fast_state_domain'][0])+np.log(pc['fast_state_domain'][1]/pc['fast_state_domain'][0])*u[:,3])
    scale=pc['scale_domain'][0]+np.ptp(pc['scale_domain'])*u[:,4]
    p=np.tile(c['published_double_slow_first'],(n,1)).astype(float)
    p[:,[1,6]]*=scale[:,None];p[:,[2,7]]*=np.sqrt(scale[:,None]);p[:,4]=vs;p[:,9]=vf
    return np.column_stack([x,vs,vf,tau]),p


def teacher_data():
    c=require_exact();dest=OUT/'teacher';dest.mkdir(exist_ok=False)
    for label,key in [('train','teacher_points'),('development','development_points')]:
        coords,p=sample(c,c['pinn'][key],c['teacher_seeds'][label]);prices,diag=batch_teacher(p,coords)
        vols=iv(prices,coords[:,0],coords[:,-1]);valid=np.isfinite(vols)
        np.savez_compressed(dest/f'{label}.npz',coords=coords,params=p,price=prices,iv=vols,iv_valid=valid)
        save(dest/f'{label}_audit.json',{**diag,'points':len(prices),'invalid_IV_only':int((~valid).sum()),'all_price_targets_retained':True})
        print('Teacher',label,len(prices),diag,flush=True)
    coords,p=sample(c,c['pinn']['collocation_points'],c['teacher_seeds']['collocation'])
    np.savez_compressed(dest/'collocation.npz',coords=coords,params=p)


def model(c):return TorchRegularVariancePINN(factors=2,width=c['pinn']['width'],depth=c['pinn']['depth'],tau_min=7/365,tau_max=2.,x_half_width=.36)
def structural(p):return tensor(np.stack([p[:,0:4],p[:,5:9]],axis=1))


def train(seed):
    c=require_exact();pc=c['pinn'];assert seed in pc['seeds'];torch.set_num_threads(1);torch.manual_seed(seed)
    dest=OUT/f'pinn_s{seed}';dest.mkdir(exist_ok=False)
    net=model(c)
    with torch.no_grad():net.head.weight.zero_();net.head.bias.zero_()
    a=np.load(OUT/'teacher/train.npz');col=np.load(OUT/'teacher/collocation.npz')
    coords,st,price=tensor(a['coords']),structural(a['params']),tensor(a['price'])
    valid=torch.tensor(a['iv_valid']); vol=tensor(np.where(a['iv_valid'],a['iv'],0.))
    cc,cs=tensor(col['coords']),structural(col['params'])
    def loss(di,ci):
        s=st[di];z=coords[di];pred=net.price(z,s)*torch.exp(-z[:,0]);pv=net.iv(z,s)
        good=valid[di]; ivloss=(pv[good]-vol[di][good]).square().mean()
        eq,dg=residual(net,cc[ci],cs[ci])
        parts=torch.stack([(pred-price[di]).square().mean(),ivloss,eq.square().mean(),torch.relu(-dg['convexity']).square().mean()])
        return parts[0]/1e-6+parts[1]/1e-4+.1*parts[2]/1e-4+.1*parts[3],parts
    opt=torch.optim.Adam(net.parameters(),lr=pc['adam_lr'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=pc['adam_steps'],eta_min=1e-5)
    started=time.monotonic();history=[]
    for step in range(pc['adam_steps']):
        di=torch.randint(len(coords),(pc['label_batch'],));ci=torch.randint(len(cc),(pc['pde_batch'],))
        opt.zero_grad(set_to_none=True);value,parts=loss(di,ci)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite PINN loss')
        value.backward();opt.step();sched.step()
        if step%500==0 or step==pc['adam_steps']-1:
            row={'step':step+1,'loss':float(value.detach()),'parts':parts.detach().tolist(),'seconds':time.monotonic()-started}
            history.append(row);save(dest/'progress.json',row);print(seed,row,flush=True)
    di=torch.arange(pc['lbfgs_labels']);ci=torch.arange(pc['lbfgs_pde'])
    opt=torch.optim.LBFGS(net.parameters(),max_iter=pc['lbfgs_steps'],line_search_fn='strong_wolfe',history_size=30)
    def closure():
        opt.zero_grad(set_to_none=True);value,_=loss(di,ci);value.backward();return value
    opt.step(closure)
    torch.save(net.state_dict(),dest/'weights.pt')
    save(dest/'completed.json',{'utc':stamp(),'seconds':time.monotonic()-started,'history':history,
        'weights_sha256':sha(dest/'weights.pt'),'manifest_sha256':sha(OUT/'manifest.json'),
        'market_weight_updates':False,'structural_parameters_learned':False})
    print('TRAINING COMPLETE',seed,flush=True)


def cases(c):
    rows=[];P=np.array(c['published_double_slow_first']);rng=np.random.default_rng(93107)
    for family,states in c['representative_states_fast_slow'].items():
        for j,(vf,vs) in enumerate(states):
            p=P.copy();p[4]=vs;p[9]=vf
            rows.append({'id':f'{family}_representative_{j}','family':family,'kind':'representative','params':p.tolist()})
    for family in c['families']:
        for j in range(c['independent_surfaces_per_family']):
            scale=float(rng.uniform(.8,3.));p=scaled(P,scale);vf,vs=.0252,.0003
            if family=='FAST_SHOCK':vf=float(rng.uniform(.0252,.06))
            if family=='SLOW_SHOCK':vs=float(rng.uniform(.0003,.01))
            if family=='FIXED_TOTAL_TWIST':vf=.04*float(rng.uniform(.125,.875));vs=.04-vf
            if family=='SMIRK_WEIGHTS':total=float(rng.uniform(.02,.06));vf=total*float(rng.uniform(.1,.9));vs=total-vf
            p[4]=vs*scale;p[9]=vf*scale;admissible(p)
            rows.append({'id':f'{family}_random_{j:02d}','family':family,'kind':'independent','params':p.tolist()})
    return rows


def surfaces():
    c=require_exact();dest=OUT/'surfaces';dest.mkdir(exist_ok=False);gr=c['controlled_grid']
    k=np.linspace(*gr['strike_over_forward'],gr['strike_nodes']);t=np.geomspace(7/365,2.,gr['tau_nodes'])
    it,ik=np.meshgrid(np.arange(len(t)),np.arange(len(k)),indexing='ij')
    x=-np.log(k[ik.ravel()]);tau=t[it.ravel()];cal=((ik.ravel()//3+it.ravel()//2)%2==0)
    rows=cases(c);save(OUT/'controlled_cases.json',rows)
    for row in rows:
        audit=[];y=exact(row['params'],x,tau,audit)
        np.savez_compressed(dest/(row['id']+'.npz'),x=x,tau=tau,calibration=cal,price=y,iv=iv(y,x,tau),strike_index=ik.ravel(),maturity_index=it.ravel())
        if audit:save(dest/(row['id']+'_fallbacks.json'),audit)
    print('CONTROLLED SURFACES',len(rows),'quotes each',len(x),flush=True)


def baselines(shard,shards):
    c=require_exact();dest=OUT/'baselines';dest.mkdir(exist_ok=True)
    for index,row in enumerate(read(OUT/'controlled_cases.json')):
        if index%shards!=shard:continue
        path=dest/(row['id']+'.json')
        if path.exists():continue
        a=np.load(OUT/'surfaces'/(row['id']+'.npz'));mask=a['calibration'];x,t,y=(a[k][mask] for k in ['x','tau','price'])
        start=time.monotonic();sh=fit_sh(x,t,y,c['single_fit'])
        inner=np.arange(len(x))%5!=1; trials=[]
        for term in [False,True]:
            fit=fit_bs(x[inner],t[inner],y[inner],term)
            score=float(np.mean((bs_predict(fit,x[~inner],t[~inner])-y[~inner])**2))
            trials.append({'term':term,'inner_MSE':score})
        best=min(trials,key=lambda r:r['inner_MSE'])['term']
        save(path,{'case':row['id'],'SH':sh,'BS_FLAT':fit_bs(x,t,y,False),'BS_TERM':fit_bs(x,t,y,True),
            'BS_selected':'BS_TERM' if best else 'BS_FLAT','BS_inner_trials':trials,'seconds':time.monotonic()-start,
            'surface_sha256':sha(OUT/'surfaces'/(row['id']+'.npz')),'manifest_sha256':sha(OUT/'manifest.json')})
        print('BASELINE',index,row['id'],'MSE',sh['best']['objective'],'near-best',sh['converged_near_best'],'sec',time.monotonic()-start,flush=True)


def nets(c):
    result=[]
    for seed in c['pinn']['seeds']:
        path=OUT/f'pinn_s{seed}';done=read(path/'completed.json')
        assert done['weights_sha256']==sha(path/'weights.pt') and done['manifest_sha256']==sha(OUT/'manifest.json')
        net=model(c);net.load_state_dict(torch.load(path/'weights.pt',weights_only=True));net.eval();result.append(net)
    return result


def neural(models,p,x,tau):
    p=np.asarray(p)
    if p.ndim==1:p=np.tile(p,(len(x),1))
    z=np.column_stack([x,p[:,4],p[:,9],tau]);pred=[]
    with torch.no_grad():
        for net in models:
            chunks=[]
            for start in range(0,len(x),2048):
                q=tensor(z[start:start+2048]);s=structural(p[start:start+2048])
                chunks.append((net.price(q,s)*torch.exp(-q[:,0])).numpy())
            pred.append(np.concatenate(chunks))
    return np.mean(pred,axis=0),pred


def metrics(y,pred,x,tau):
    e=pred-y;abs_e=abs(e);a=iv(y,x,tau);b=iv(pred,x,tau);valid=np.isfinite(a)&np.isfinite(b)
    return {'price_RMSE':float(np.sqrt(np.mean(e*e))),'price_MAE':float(abs_e.mean()),'price_P95':float(np.quantile(abs_e,.95)),
        'price_max':float(abs_e.max()),'forward_normalized_price_RMSE':float(np.sqrt(np.mean(e*e))),
        'IV_RMSE_volatility_points':float(100*np.sqrt(np.mean((a[valid]-b[valid])**2))) if valid.any() else None,
        'IV_valid_quotes':int(valid.sum()),'quotes':len(y)}


def lock():
    c=require_exact();nets(c)
    for case in read(OUT/'controlled_cases.json'):assert (OUT/'baselines'/(case['id']+'.json')).exists()
    assert not (OUT/'controlled_results.json').exists()
    save(OUT/'evaluation_lock.json',{'utc':stamp(),'manifest_sha256':sha(OUT/'manifest.json'),
        'models':{str(s):sha(OUT/f'pinn_s{s}/weights.pt') for s in c['pinn']['seeds']},
        'baselines':{p.name:sha(p) for p in sorted((OUT/'baselines').glob('*.json'))},
        'market_selection_sha256':sha(OUT/'market_selection.json') if (OUT/'market_selection.json').exists() else None})
    print('ALL CHECKPOINTS/BASELINES LOCKED BEFORE HEADLINE SCORING',flush=True)


def evaluate():
    c=require_exact();assert (OUT/'evaluation_lock.json').exists();assert not (OUT/'controlled_results.json').exists()
    models=nets(c);pc=c['pinn'];coords,p=sample(c,pc['fidelity_test_points'],c['teacher_seeds']['fidelity'])
    y,diag=batch_teacher(p,coords);pred,seeds=neural(models,p,coords[:,0],coords[:,-1]);fidelity=metrics(y,pred,coords[:,0],coords[:,-1])
    gates=pc['fidelity_gate']
    fidelity['pass']=bool(fidelity['price_RMSE']<=gates['forward_price_RMSE_max'] and fidelity['price_P95']<=gates['forward_price_P95_max'] and fidelity['price_max']<=gates['forward_price_max_max'] and fidelity['IV_RMSE_volatility_points'] is not None and fidelity['IV_RMSE_volatility_points']/100<=gates['IV_RMSE_max'])
    fidelity['seed_metrics']=[metrics(y,z,coords[:,0],coords[:,-1]) for z in seeds];fidelity['teacher_numerics']=diag
    ff=pd.DataFrame({'x':coords[:,0],'v_slow':coords[:,1],'v_fast':coords[:,2],'tau':coords[:,-1],
                  'teacher':y,'pinn':pred,'residual':pred-y})
    for j in range(10):ff[f'parameter_{j}']=p[:,j]
    for seed,z in zip(pc['seeds'],seeds):ff[f'pinn_s{seed}']=z
    ff.to_csv(OUT/'fidelity_predictions.csv',index=False)
    fresh,fp=sample(c,pc['pde_test_points'],c['teacher_seeds']['pde_fidelity']);pde=[]
    for net in models:
        eq=[]
        for j in range(0,len(fresh),128):eq.append(residual(net,tensor(fresh[j:j+128]),structural(fp[j:j+128]))[0].detach().numpy())
        pde.append(float(np.sqrt(np.mean(np.concatenate(eq)**2))))
    fidelity['fresh_scaled_PDE_RMSE_by_seed']=pde
    records=[];predictions=[];buckets=[]
    for case in read(OUT/'controlled_cases.json'):
        data=OUT/'surfaces'/(case['id']+'.npz');a=np.load(data);base=read(OUT/'baselines'/(case['id']+'.json'))
        assert base['surface_sha256']==sha(data)
        x,t,y=a['x'],a['tau'],a['price'];test=~a['calibration'];pinn,seed_predictions=neural(models,case['params'],x,t)
        estimates={'DH_EXACT_TEACHER':y,'DH_PINN':pinn,'SH_BEST_FOUND':exact(base['SH']['best']['params'],x,t),
            'BS_FLAT':bs_predict(base['BS_FLAT'],x,t),'BS_TERM':bs_predict(base['BS_TERM'],x,t)}
        estimates['BS_SELECTED']=estimates[base['BS_selected']]
        for name,z in estimates.items():
            records.append({'case':case['id'],'family':case['family'],'kind':case['kind'],'model':name,**metrics(y[test],z[test],x[test],t[test]),
                            'SH_near_best_starts':base['SH']['converged_near_best']})
            for label,m in [('short',t<=30/365),('medium', (t>30/365)&(t<=90/365)),('long',(t>90/365)&(t<=1)),('very_long',t>1),
                            ('ATM',abs(np.exp(-x)-1)<=.02),('wing',(np.exp(-x)<.9)|(np.exp(-x)>1.1))]:
                m=m&test
                if m.any():buckets.append({'case':case['id'],'model':name,'bucket':label,**metrics(y[m],z[m],x[m],t[m])})
            for lo,hi in zip([.7,.9,.98,1.02,1.1],[.9,.98,1.02,1.1,1.30000001]):
                m=test&(np.exp(-x)>=lo)&(np.exp(-x)<hi)
                if m.any():buckets.append({'case':case['id'],'model':name,'bucket':f'K/F_{lo:.2f}_{hi:.2f}',**metrics(y[m],z[m],x[m],t[m])})
        frame=pd.DataFrame({'case':case['id'],'family':case['family'],'kind':case['kind'],'x':x,'strike_over_forward':np.exp(-x),'tau':t,'calibration':~test,
            'maturity_index':a['maturity_index'],'strike_index':a['strike_index'],**estimates})
        for name,z in estimates.items():frame[name+'_residual']=z-y
        for seed,z in zip(pc['seeds'],seed_predictions):frame[f'DH_PINN_s{seed}']=z
        frame['advantage']=abs(estimates['SH_BEST_FOUND']-y)-abs(pinn-y)
        predictions.append(frame)
    pd.concat(predictions).to_csv(OUT/'controlled_predictions.csv',index=False)
    table=pd.DataFrame(records);table.to_csv(OUT/'controlled_metrics.csv',index=False)
    pd.DataFrame(buckets).to_csv(OUT/'bucket_metrics.csv',index=False)
    rng=np.random.default_rng(c['statistics']['seed']);stats=[]
    for family in c['families']:
        a=table[(table.family==family)&(table.kind=='independent')].pivot(index='case',columns='model',values='price_RMSE')
        delta=(a.SH_BEST_FOUND-a.DH_PINN).to_numpy();draw=rng.choice(delta,size=(c['statistics']['bootstrap_replicates'],len(delta)),replace=True).mean(1)
        ratio=float(np.sqrt(np.mean(a.DH_PINN**2))/np.sqrt(np.mean(a.SH_BEST_FOUND**2)))
        ci=np.quantile(draw,[.005,.025,.975,.995]);stats.append({'family':family,'surfaces':len(a),'mean_advantage':float(delta.mean()),'median_advantage':float(np.median(delta)),
            'DH_win_fraction':float((delta>0).mean()),'hierarchy_DH_SH_BS_fraction':float(((a.DH_PINN<a.SH_BEST_FOUND)&(a.SH_BEST_FOUND<a.BS_SELECTED)).mean()),
            'paired_bootstrap95':ci[1:3].tolist(),'paired_bootstrap99_bonferroni':ci[[0,3]].tolist(),'DH_over_SH_pooled_RMSE':ratio,
            'reliable_structural_advantage':bool(fidelity['pass'] and ratio<=.2 and ci[0]>0)})
    save(OUT/'controlled_results.json',{'utc':stamp(),'fidelity':fidelity,'statistics':stats,'no_retraining_after_test':True,'exact_teacher_zero_is_not_evidence':True})
    print(json.dumps({'fidelity':fidelity,'statistics':stats},indent=2),flush=True)


def market_final():
    c=require_exact();a=market_panel('final')
    if a is None or not (OUT/'market_selection.json').exists():print('Final market test blocked, no substitution',flush=True);return
    assert a.date.nunique()>=10; assert not (OUT/'market_final_predictions.csv').exists()
    selected=read(OUT/'market_selection.json')['selected'];models=nets(c);frames=[];state_diagnostics=[]
    for date,g in a.groupby('date'):
        cal=g[g.split.eq('anchor')];h=g[g.split.eq('test')];x,t=h.x.to_numpy(),h.tau.to_numpy()
        y=h.market_call_normalized.to_numpy()*np.exp(-x);f=h.copy();f['target_forward_call']=y
        for kind in ['single','double']:
            p=np.array(selected[kind]['params']);ref=exact(p,x,t);f[kind+'_fixed_exact']=ref
            if kind=='double':f['double_fixed_PINN']=neural(models,p,x,t)[0]
            cx,ct=cal.x.to_numpy(),cal.tau.to_numpy();cy=cal.market_call_normalized.to_numpy()*np.exp(-cx)
            grid=Grid(cx,ct);indices=[4] if kind=='single' else [4,9];bound=np.log(c['state_adaptive']['state_bounds'])
            starts=qmc.LatinHypercube(len(indices),seed=93123).random(c['state_adaptive']['starts'])
            records=[]
            for j,u in enumerate(starts):
                def make(z):q=p.copy();q[indices]=np.exp(z);return q
                r=least_squares(lambda z:(grid(make(z))-cy)/.01,bound[0]+np.ptp(bound)*u,bounds=bound,max_nfev=c['state_adaptive']['max_nfev'])
                records.append({'start':j,'params':make(r.x).tolist(),'objective':float(np.mean((r.fun*.01)**2)),'success':bool(r.success)})
            best=min(records,key=lambda z:z['objective']);f[kind+'_state_exact']=exact(best['params'],x,t)
            if kind=='double':
                q=np.array(best['params']);inside=c['pinn']['slow_state_domain'][0]<=q[4]<=c['pinn']['slow_state_domain'][1] and c['pinn']['fast_state_domain'][0]<=q[9]<=c['pinn']['fast_state_domain'][1]
                f['double_state_PINN']=neural(models,q,x,t)[0] if inside else np.nan
            state_diagnostics.append({'date':date,'model':kind,'label':c['state_adaptive']['label'],'starts':records,'selected':best})
        f['BS_fixed']=bs_predict(selected['bs'],x,t)
        for term in [False,True]:
            fit=fit_bs(cal.x.to_numpy(),cal.tau.to_numpy(),cal.market_call_normalized.to_numpy()*np.exp(-cal.x.to_numpy()),term)
            f['BS_state_term' if term else 'BS_state_flat']=bs_predict(fit,x,t)
        frames.append(f)
    pd.concat(frames).to_csv(OUT/'market_final_predictions.csv',index=False);save(OUT/'market_state_diagnostics.json',state_diagnostics)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','validate','fetch-validation','select-market','teacher','surfaces','baselines','train','lock','evaluate','fetch-final','market-final','report'])
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args();torch.set_num_threads(1)
    if args.command=='freeze':freeze()
    elif args.command=='validate':validate()
    elif args.command=='fetch-validation':fetch_market('validation')
    elif args.command=='select-market':market_select()
    elif args.command=='teacher':teacher_data()
    elif args.command=='surfaces':surfaces()
    elif args.command=='baselines':baselines(args.shard,args.shards)
    elif args.command=='train':train(args.seed)
    elif args.command=='lock':lock()
    elif args.command=='evaluate':evaluate()
    elif args.command=='fetch-final':fetch_market('final')
    elif args.command=='market-final':market_final()
    elif args.command=='report':
        from report import main
        main()
