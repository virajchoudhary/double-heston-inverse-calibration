"""Separate, frozen-parameter forward test. Never fits structural parameters.

Reuses the regular price-PDE PINN and independent numerical Fourier engine.
Commands deliberately separate source preparation, synthetic-only training and
market scoring. No change to the V2 inverse-calibration experiment is made.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mentor_dh_pinn.audit_crisis_option_coverage import parity_gate
from src.mentor_dh_pinn.regular_pinn_data import invert_total_variance
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual
from src.mentor_dh_pinn.torch_pricer import price_call, price_call_single


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tensor(a):
    return torch.as_tensor(a, dtype=torch.float64)


def single_match(p):
    f = np.asarray(p, dtype=float).reshape(2, 5)
    k, t, s, r, v = f.T
    total, theta = v.sum(), t.sum()
    if abs(total - theta) < 1e-12:
        raise ValueError('Initial drift matching is undefined when v0 equals theta')
    sigma = np.sqrt(np.sum(s*s*v)/total)
    single = np.array([np.sum(k*(v-t))/(total-theta), theta, sigma,
                       np.sum(r*s*v)/(sigma*total), total])
    assert single[0] > 0 and abs(single[3]) < 1
    return single


def parameters(cfg, model):
    p = np.array(cfg['double_parameters_slow_then_fast'])
    return p if model == 'double' else single_match(p)


def state(xt, p):
    v = np.asarray(p).reshape(-1, 5)[:, 4]
    return tensor(np.column_stack([xt[:, 0], np.tile(v, (len(xt), 1)), xt[:, 1]]))


def reference(p, xt, nodes=128):
    engine = price_call if len(p) == 10 else price_call_single
    chunks = []
    with torch.no_grad():
        for a in np.array_split(xt, max(1, int(np.ceil(len(xt)/512)))):
            x, tau = tensor(a[:, 0]), tensor(a[:, 1])
            chunks.append(engine(tensor(p), x.exp(), torch.ones_like(x), tau,
                                 torch.zeros_like(x), torch.zeros_like(x), node_count=nodes).numpy())
    return np.concatenate(chunks)


def lhs_xt(cfg, n, seed):
    u = qmc.LatinHypercube(2, seed=seed).random(n)
    return qmc.scale(u, [cfg['x_bounds'][0], cfg['tau_days'][0]/365],
                     [cfg['x_bounds'][1], cfg['tau_days'][1]/365])


def collocation(cfg, factors, n, seed):
    # Same total-variance domain for both arms; split it only for Double Heston.
    u = qmc.LatinHypercube(4, seed=seed).random(n)
    x = cfg['x_bounds'][0] + np.ptp(cfg['x_bounds'])*u[:, 0]
    tau = (cfg['tau_days'][0] + np.ptp(cfg['tau_days'])*u[:, 1])/365
    total = cfg['total_variance_state_bounds'][0] + np.ptp(cfg['total_variance_state_bounds'])*u[:, 2]
    if factors == 1:
        return tensor(np.column_stack([x, total, tau]))
    share = cfg['double_variance_share_bounds'][0] + np.ptp(cfg['double_variance_share_bounds'])*u[:, 3]
    return tensor(np.column_stack([x, total*share, total*(1-share), tau]))


def normalized_call(otm, option, x, discount, strike):
    return otm/(discount*strike) + np.where(np.asarray(option) == 'PE', np.expm1(x), 0.)


def raw_option(call, option, x, discount, strike):
    return discount*strike*(call - np.where(np.asarray(option) == 'PE', np.expm1(x), 0.))


def prepare(cfg, workspace, out, config_path):
    out.mkdir(parents=True, exist_ok=False)
    deps = [Path(__file__), config_path,
            ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',
            ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',
            ROOT/'src/mentor_dh_pinn/torch_pricer.py',
            ROOT/'scripts/mentor_dh_pinn/audit_crisis_option_coverage.py']
    write_json(out/'protocol.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg, 'files': {str(p): sha(p) for p in deps},
        'models': {m: parameters(cfg, m).tolist() for m in ['single', 'double']}})
    manifest = workspace/'outputs/019fc8a0/nse_source_manifest.csv'
    entries = pd.read_csv(manifest)
    audit = {'source_manifest_sha256': sha(manifest), 'source_files': [], 'windows': {},
             'provenance_scope': 'Archived official-URL manifest and local ZIP hash verification; not a fresh remote download verification'}
    rows, groups, rejects = [], [], []
    for window, (lo, hi) in cfg['windows'].items():
        selected = entries[entries.date.between(lo, hi)]
        count = Counter()
        for item in selected[selected.status.eq('downloaded')].itertuples():
            path = workspace/item.file
            if sha(path) != item.sha256:
                raise ValueError(f'Source hash mismatch: {path}')
            audit['source_files'].append({'file': item.file, 'url': item.url, 'sha256': item.sha256})
            with zipfile.ZipFile(path) as z:
                csv = [n for n in z.namelist() if n.lower().endswith('.csv')]
                assert len(csv) == 1
                with z.open(csv[0]) as f:
                    raw = pd.read_csv(f)
            legacy = 'SYMBOL' in raw
            if legacy:
                raw = raw[raw.INSTRUMENT.eq('OPTIDX') & raw.SYMBOL.eq('NIFTY')]
                mapping = dict(SYMBOL='symbol', TIMESTAMP='date', EXPIRY_DT='expiry', STRIKE_PR='strike',
                    OPTION_TYP='option', CLOSE='price', CONTRACTS='contracts', OPEN_INT='oi',
                    OPEN='open', HIGH='high', LOW='low')
            else:
                raw = raw[raw.FinInstrmTp.eq('IDO') & raw.TckrSymb.eq('NIFTY')]
                mapping = dict(TckrSymb='symbol', TradDt='date', FininstrmActlXpryDt='expiry', StrkPric='strike',
                    OptnTp='option', ClsPric='price', TtlTradgVol='contracts', OpnIntrst='oi',
                    OpnPric='open', HghPric='high', LwPric='low')
            a = raw.rename(columns=mapping)[list(mapping.values())].copy()
            count['listed_rows'] += len(a)
            count['exact_duplicate_rows_removed'] += int(a.duplicated().sum())
            a = a.drop_duplicates()
            a['date'] = pd.to_datetime(a.date, format='mixed', errors='coerce')
            a['expiry'] = pd.to_datetime(a.expiry, format='mixed', errors='coerce')
            assert a.date.eq(pd.Timestamp(item.date)).all(), 'Trade date mismatch'
            assert not a.duplicated(['date', 'expiry', 'strike', 'option']).any(), 'Conflicting keys'
            num = ['strike','price','contracts','oi','open','high','low']
            a[num] = a[num].apply(pd.to_numeric, errors='coerce')
            basic = (np.isfinite(a[num]).all(axis=1) & a.expiry.notna() & a.option.isin(['CE','PE'])
                     & a[num].gt(0).all(axis=1) & a.price.ge(a.low-1e-8)
                     & a.price.le(a.high+1e-8) & a.open.ge(a.low-1e-8) & a.open.le(a.high+1e-8))
            for r in a.loc[~basic].itertuples():
                rejects.append({'window':window,'date':item.date,'expiry':str(r.expiry),'strike':r.strike,
                                'option':r.option,'reason':'nonfinite/nonpositive/date/option/OHLC gate'})
            count['basic_rejected_rows'] += int((~basic).sum())
            a = a[basic]
            for expiry, b in a.groupby('expiry'):
                tau = (expiry-pd.Timestamp(item.date)).days/365
                g = {'window':window,'date':item.date,'expiry':str(expiry.date()), 'tau':tau}
                if not cfg['tau_days'][0]/365 <= tau <= cfg['tau_days'][1]/365:
                    groups.append({**g,'status':'outside_tau'}); continue
                pairs = b.pivot(index='strike', columns='option', values='price').reindex(columns=['CE','PE']).dropna().sort_index()
                if len(pairs) < cfg['market_filters']['minimum_pairs']:
                    groups.append({**g,'status':'insufficient_pairs'}); continue
                carry = parity_gate(pairs, tau)
                if carry is None:
                    groups.append({**g,'status':'anchor_parity_failed'}); continue
                F, D = carry['forward_from_anchor_parity'], carry['discount_from_anchor_parity']
                for j, (K, pair) in enumerate(pairs.iterrows()):
                    opt = 'CE' if K >= F else 'PE'
                    price, x = float(pair[opt]), float(np.log(F/K))
                    c = float(normalized_call(price,opt,x,D,K))
                    w = float(invert_total_variance(np.array([c]),np.array([x]))[0])
                    iv = np.sqrt(w/tau) if np.isfinite(w) else float('nan')
                    d1 = x/np.sqrt(w)+np.sqrt(w)/2 if w > 0 else float('nan')
                    vega = np.sqrt(tau)*np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)
                    valid = (cfg['x_bounds'][0] <= x <= cfg['x_bounds'][1] and np.isfinite(iv)
                             and cfg['market_filters']['IV_bounds'][0] <= iv <= cfg['market_filters']['IV_bounds'][1]
                             and vega >= cfg['market_filters']['minimum_vega_over_DF'])
                    if not valid:
                        rejects.append({**g,'strike':K,'option':opt,'reason':'moneyness/arbitrage/IV/vega gate'}); continue
                    rows.append({**g,'strike':K,'option':opt,'price':price,'x':x,'forward':F,'discount':D,
                        'market_call_normalized':c,'market_iv':iv,'vega_over_DF':vega,
                        'split':'test' if j%3 == 1 else 'anchor','source_sha256':item.sha256})
                groups.append({**g,'status':'carry_accepted',**carry})
        audit['windows'][window] = {'counts':dict(count), 'unavailable':selected.loc[~selected.status.eq('downloaded'),['date','status']].to_dict('records')}
    panel = pd.DataFrame(rows)
    ntest = panel[panel.split.eq('test')].groupby('date').size()
    dates = ntest[ntest.ge(cfg['market_filters']['minimum_holdout_quotes_per_date'])].index
    panel['date_eligible'] = panel.date.isin(dates)
    assert np.isfinite(panel.select_dtypes('number')).all().all()
    assert not panel.duplicated(['date','expiry','strike','option']).any()
    panel.to_csv(out/'clean_quotes.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'expiry_group_audit.csv',index=False)
    pd.DataFrame(rejects).to_csv(out/'rejections.csv',index=False)
    audit['clean_counts'] = panel.groupby(['window','split','date_eligible']).size().rename('rows').reset_index().to_dict('records')
    audit['clean_quotes_sha256'] = sha(out/'clean_quotes.csv')
    audit['expiry_handling'] = 'Legacy EXPIRY_DT / UDiFF FininstrmActlXpryDt, ACT/365; never force expiry weekday or round to months'
    audit['corporate_actions'] = 'NIFTY index options, not individual stock options. Preserve raw index/strike levels; no split-adjustment of index option prices or guessed dividends.'
    write_json(out/'data_audit.json',audit)
    print(json.dumps(audit['clean_counts'],indent=2),flush=True)


def verify(out):
    protocol = json.loads((out/'protocol.json').read_text())
    for path, expected in protocol['files'].items():
        assert sha(path) == expected, f'Frozen implementation changed: {path}'
    return protocol['config']


def train(out, model_name, seed):
    cfg = verify(out)
    assert seed in cfg['seeds']
    dest = out/f'{model_name}_s{seed}'
    dest.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    p = parameters(cfg,model_name)
    structural = tensor(p.reshape(-1,5)[:,:4])
    assert not structural.requires_grad
    factors = len(p)//5
    net = TorchRegularVariancePINN(factors=factors,width=cfg['width'],depth=cfg['depth'],
        tau_min=cfg['tau_days'][0]/365,tau_max=cfg['tau_days'][1]/365,x_half_width=.35)
    xt = lhs_xt(cfg,cfg['synthetic_training_points'],91201)
    y = reference(p,xt)
    quadrature_error = float(np.max(abs(y-reference(p,xt,96))))
    assert quadrature_error <= cfg['quadrature_max_difference']
    iv = np.sqrt(invert_total_variance(y,xt[:,0])/xt[:,1])
    assert np.isfinite(iv).all()
    coords, y, iv = state(xt,p), tensor(y), tensor(iv)
    coll = collocation(cfg,factors,cfg['collocation_points'],91202)
    np.savez_compressed(dest/'training_inputs.npz',xt=xt,price=y.numpy(),iv=iv.numpy(),collocation=coll.numpy(),fixed_parameters=p)
    opt = torch.optim.Adam(net.parameters(),lr=cfg['adam_learning_rate'])
    history = []
    start = time.monotonic()
    def loss(di, ci):
        pred_iv = net.iv(coords[di],structural)
        pred_c = net.price(coords[di],structural)
        eq, diag = residual(net,coll[ci],structural)
        parts = torch.stack([(pred_c-y[di]).square().mean(),(pred_iv-iv[di]).square().mean(),
                             eq.square().mean(),torch.relu(-diag['convexity']).square().mean()])
        return parts[:3].sum()/1e-4 + .1*parts[3], parts
    for step in range(cfg['adam_steps']):
        di = torch.randint(len(coords),(cfg['label_batch'],))
        ci = torch.randint(len(coll),(cfg['pde_batch'],))
        opt.zero_grad(set_to_none=True)
        value,parts = loss(di,ci)
        assert torch.isfinite(value)
        value.backward(); opt.step()
        if step%200 == 0 or step == cfg['adam_steps']-1:
            entry = {'phase':'adam','step':step+1,'loss':float(value.detach()),'parts':parts.detach().tolist(),'seconds':time.monotonic()-start}
            history.append(entry); print(model_name,seed,entry,flush=True)
            write_json(dest/'progress.json',entry)
    di = torch.arange(cfg['lbfgs_label_points'])
    ci = torch.arange(cfg['lbfgs_pde_points'])
    opt = torch.optim.LBFGS(net.parameters(),max_iter=cfg['lbfgs_steps'],lr=1.,line_search_fn='strong_wolfe',history_size=30)
    def closure():
        opt.zero_grad(set_to_none=True)
        value,_ = loss(di,ci)
        assert torch.isfinite(value)
        value.backward()
        return value
    opt.step(closure)
    assert not structural.requires_grad and structural.grad is None
    assert np.array_equal(p,parameters(cfg,model_name))
    torch.save(net.state_dict(),dest/'weights.pt')
    write_json(dest/'training.json',{'status':'completed','seconds':time.monotonic()-start,'fixed_parameters':p.tolist(),
        'only_network_weights_optimized':True,'training_source':'synthetic Fourier labels only; market file never opened by train()',
        'quadrature_max_difference':quadrature_error,'history':history,'torch_version':torch.__version__,
        'protocol_sha256':sha(out/'protocol.json'),'weights_sha256':sha(dest/'weights.pt')})
    print(model_name,seed,'COMPLETE',time.monotonic()-start,flush=True)


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def score(out):
    cfg = verify(out)
    audit = json.loads((out/'data_audit.json').read_text())
    assert sha(out/'clean_quotes.csv') == audit['clean_quotes_sha256']
    panel = pd.read_csv(out/'clean_quotes.csv')
    market = panel[panel.split.eq('test') & panel.date_eligible].copy()
    xt = market[['x','tau']].to_numpy()
    held = lhs_xt(cfg,cfg['synthetic_test_points'],91301)
    diagnostics = []
    torch.set_num_threads(1)
    for model_name in ['single','double']:
        p = parameters(cfg,model_name)
        structural = tensor(p.reshape(-1,5)[:,:4])
        exact = reference(p,xt)
        quadrature_error = float(max(np.max(abs(exact-reference(p,xt,96))),
                                     np.max(abs(reference(p,held)-reference(p,held,96)))))
        assert quadrature_error <= cfg['quadrature_max_difference']
        market[model_name+'_reference'] = raw_option(exact,market.option,market.x,market.discount,market.strike)
        market[model_name+'_reference_iv'] = np.sqrt(invert_total_variance(exact,xt[:,0])/xt[:,1])
        teacher_held = reference(p,held)
        iv_held = np.sqrt(invert_total_variance(teacher_held,held[:,0])/held[:,1])
        predictions = []
        for seed in cfg['seeds']:
            dest = out/f'{model_name}_s{seed}'
            done = json.loads((dest/'training.json').read_text())
            assert done['status']=='completed' and done['protocol_sha256']==sha(out/'protocol.json')
            assert done['weights_sha256']==sha(dest/'weights.pt')
            net = TorchRegularVariancePINN(factors=len(p)//5,width=cfg['width'],depth=cfg['depth'],
                tau_min=cfg['tau_days'][0]/365,tau_max=cfg['tau_days'][1]/365,x_half_width=.35)
            net.load_state_dict(torch.load(dest/'weights.pt',weights_only=True)); net.eval()
            with torch.no_grad():
                c = net.price(state(xt,p),structural).numpy()
                synthetic_iv = net.iv(state(held,p),structural).numpy()
            predictions.append(c)
            market[f'{model_name}_s{seed}'] = raw_option(c,market.option,market.x,market.discount,market.strike)
            fresh = collocation(cfg,len(p)//5,cfg['fresh_pde_test_points'],91302)
            eqs = []
            for chunk in fresh.split(128):
                eq,_ = residual(net,chunk,structural); eqs.append(eq.detach().numpy())
            diag = {'model':model_name,'seed':seed,'synthetic_IV_RMSE':rmse(synthetic_iv-iv_held),
                    'market_price_RMSE_over_DF':rmse((c-exact)/np.exp(xt[:,0])),
                    'fresh_scaled_PDE_RMSE':rmse(np.concatenate(eqs)), 'quadrature_max_difference':quadrature_error}
            diag['fidelity_pass'] = all(diag[k.removesuffix('_max')] <= limit for k,limit in cfg['neural_fidelity_gate'].items())
            diagnostics.append(diag)
        ensemble = np.mean(predictions,axis=0)
        market[model_name+'_pinn'] = raw_option(ensemble,market.option,market.x,market.discount,market.strike)
        market[model_name+'_pinn_iv'] = np.sqrt(invert_total_variance(ensemble,xt[:,0])/xt[:,1])
    assert np.isfinite(market.select_dtypes('number')).all().all()
    market.to_csv(out/'test_predictions.csv',index=False)
    results, daily = [], []
    rng = np.random.default_rng(91217)
    for window, block in market.groupby('window',sort=False):
        for engine in ['reference','pinn']:
            errors = {}
            for model in ['single','double']:
                err = (block[model+'_'+engine]-block.price)/(block.discount*block.forward)
                date_mse = err.square() if hasattr(err,'square') else err**2
                errors[model] = date_mse.groupby(block.date).mean()
                for date,value in errors[model].items():
                    daily.append({'window':window,'engine':engine,'model':model,'date':date,'normalized_MSE':float(value)})
            delta = (errors['double']-errors['single']).to_numpy()
            boot = []
            for _ in range(2000):
                starts = rng.integers(len(delta),size=int(np.ceil(len(delta)/3)))
                idx = ((starts[:,None]+np.arange(3))%len(delta)).ravel()[:len(delta)]
                boot.append(float(delta[idx].mean()))
            results.append({'window':window,'engine':engine,'quotes':len(block),'dates':len(delta),
                'single_equal_date_NRMSE':float(np.sqrt(errors['single'].mean())),
                'double_equal_date_NRMSE':float(np.sqrt(errors['double'].mean())),
                'single_RMSE_index_points':rmse(block['single_'+engine]-block.price),
                'double_RMSE_index_points':rmse(block['double_'+engine]-block.price),
                'single_IV_RMSE':rmse(block['single_'+engine+'_iv']-block.market_iv),
                'double_IV_RMSE':rmse(block['double_'+engine+'_iv']-block.market_iv),
                'double_win_dates':int((delta < -1e-12).sum()),'tie_dates':int((abs(delta)<=1e-12).sum()),
                'mean_DH_minus_SH_normalized_MSE':float(delta.mean()),
                'paired_block_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist()})
    pd.DataFrame(daily).to_csv(out/'daily_metrics.csv',index=False)
    write_json(out/'results.json',{'neural_diagnostics':diagnostics,'comparisons':results,
        'all_neural_fidelity_gates_pass':all(d['fidelity_pass'] for d in diagnostics),
        'limitations':['Fixed published parameters are not fitted NIFTY parameters',
         'Anchor carry uses same-date closes: reconstruction, not forecasting; closes are not synchronous executable bid/ask quotes',
         'Only two training seeds and one fixed paired parameter scenario; not universal model-class superiority',
         'Bootstrap is descriptive, small-window and conditional on carry estimates; not a causal effect of war or a multiple-testing-adjusted finding',
         'Full calendar windows include non-event days; no peak-volatility ranking was performed',
         'Historical study choice and previous experiments are visible: no claim of prospectively unseen research data'],
        'test_predictions_sha256':sha(out/'test_predictions.csv')})
    plot(out,market,results)
    print(json.dumps({'diagnostics':diagnostics,'comparisons':results},indent=2),flush=True)


def plot(out, market, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    windows = list(dict.fromkeys(r['window'] for r in results))
    for ax,window in zip(axes,windows):
        r = [r for r in results if r['window']==window]
        x = np.arange(len(r))
        ax.bar(x-.18,[100*a['single_equal_date_NRMSE'] for a in r],.36,label='Single Heston')
        ax.bar(x+.18,[100*a['double_equal_date_NRMSE'] for a in r],.36,label='Double Heston')
        ax.set_xticks(x,[a['engine'] for a in r]); ax.set_title(window.replace('_',' '))
        ax.set_ylabel('Equal-date normalized RMSE (%) — lower is better'); ax.legend(fontsize=8)
    fig.suptitle('NIFTY options: fixed published DH1 vs moment-matched Single Heston\nReference prices and two-seed PINN average; identical held-out strikes')
    fig.savefig(out/'crisis_price_comparison.png',dpi=170); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,window in zip(axes,windows):
        a = market[market.window.eq(window)]
        ax.scatter(a.market_iv,a.single_pinn_iv,s=7,alpha=.3,label='Single')
        ax.scatter(a.market_iv,a.double_pinn_iv,s=7,alpha=.3,label='Double')
        lo=min(a.market_iv.min(),a.single_pinn_iv.min(),a.double_pinn_iv.min())
        hi=max(a.market_iv.max(),a.single_pinn_iv.max(),a.double_pinn_iv.max())
        ax.plot([lo,hi],[lo,hi],'k--',lw=1,label='Perfect agreement')
        ax.set(xlabel='Observed close-implied IV',ylabel='Fixed-parameter PINN IV',title=window.replace('_',' '))
        ax.legend(fontsize=8)
    fig.suptitle('Implied-volatility diagnostic — not realized volatility or an unseen-date forecast')
    fig.savefig(out/'crisis_iv_diagnostic.png',dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','train','score'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/nifty_fixed_crisis.json')
    parser.add_argument('--workspace',type=Path,default=ROOT.parent)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',choices=['single','double'])
    parser.add_argument('--seed',type=int)
    args = parser.parse_args()
    if args.command=='prepare': prepare(json.loads(args.config.read_text()),args.workspace,args.out,args.config)
    elif args.command=='train': train(args.out,args.model,args.seed)
    else: score(args.out)
