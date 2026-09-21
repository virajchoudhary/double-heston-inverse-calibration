"""Reporting-only completion after the one-time test; no model reselection.

Adds common-IV/region diagnostics, independent teacher checks and a parallel
branch schematic. This reporting source is separate from the frozen selector.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle
from .common import *
from .report import main as report_main,md
from .audit import main as integrity_audit
from experiments.nifty_multifactor_v4.literature_exact import iv,adaptive


def diagram(chosen):
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained')
    def box(ax,xy,w,h,text):
        ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=.009',facecolor='#edf3f8',edgecolor='#326a96'))
        ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#344454'})
    for ax,is_new in zip(axes,[False,True]):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        ax.set_title('Selected: '+chosen if is_new else 'Baseline: two parallel contributions to log IV')
        box(ax,(.015,.34),.20,.27,'Same dimensionless\ncoordinates + parameters\nx, τ, v_s, v_f, structural')
        box(ax,(.29,.66),.31,.22,'Frozen v5 SHARED checkpoint\nC3 + existing correction' if is_new else 'C3 core: 24 features\n5 hidden layers × 256 tanh\nδ_C3')
        box(ax,(.29,.09),.31,.25,'26 features; train-only affine scaling\nModified MLP: U,V + 5 × 96 gates\nδ_new = 0.05 tanh(head)' if is_new else '26 features (adds factor decays)\n3 shared 2-layer × 32 branches\nδ_v5 = bounded mean correction')
        arrow(ax,(.215,.54),(.29,.77));arrow(ax,(.215,.41),(.29,.22))
        ax.add_patch(Circle((.68,.48),.032,facecolor='white',edgecolor='#326a96'));ax.text(.68,.48,'+',ha='center',va='center',fontsize=18)
        arrow(ax,(.60,.77),(.66,.51));arrow(ax,(.60,.22),(.66,.45))
        box(ax,(.765,.30),.215,.36,'δ = sum of corrections\nw = τ v̄ exp(2δ)\nc = Black(x,w)\nExact payoff at τ=0')
        arrow(ax,(.712,.48),(.765,.48))
        if is_new:ax.text(.5,.015,'New head starts at zero; inherited function preserved. Branches receive inputs in parallel.',ha='center',fontsize=8)
    fig.savefig(OUT/'figures/architecture_diagrams.png',dpi=160);plt.close(fig)


def main():
    integrity_audit();selected=read(OUT/'selection.json')['chosen']
    if selected is None:report_main();return
    final=pd.read_csv(OUT/'final_fidelity/metrics.csv').set_index('arm');base=final.loc['BASELINE'];best=final.loc[selected]
    dev=pd.read_csv(OUT/'PINN_ARCHITECTURE_ABLATION.csv').set_index('arm')
    a=np.load(OUT/'final_fidelity/data.npz');x,t=a['coords'][:,0],a['coords'][:,-1];y=a['price']
    pb=np.load(OUT/'final_fidelity/BASELINE_predictions.npy');pn=np.load(OUT/f'final_fidelity/{selected}_predictions.npy')
    vy,vb,vn=iv(y,x,t),iv(pb,x,t),iv(pn,x,t);valid=np.isfinite(vy)&np.isfinite(vb)&np.isfinite(vn)
    common={'quotes':int(valid.sum()),'baseline_IV_RMSE_vol_points':float(100*np.sqrt(np.mean((vb[valid]-vy[valid])**2))),
            'selected_IV_RMSE_vol_points':float(100*np.sqrt(np.mean((vn[valid]-vy[valid])**2)))}
    save(OUT/'final_fidelity/common_IV_comparison.json',common)
    buckets={'7–30d':t<=30/365,'30–90d':(t>30/365)&(t<=90/365),'90–365d':(t>90/365)&(t<=1),'365–730d':t>1,
             'ATM |x|≤.02':abs(x)<=.02,'wings |x|>.15':abs(x)>.15}
    rows=[]
    for name,mask in buckets.items():
        eb=float(np.sqrt(np.mean((pb[mask]-y[mask])**2)));en=float(np.sqrt(np.mean((pn[mask]-y[mask])**2)))
        rows.append({'bucket':name,'quotes':int(mask.sum()),'baseline_RMSE':eb,'selected_RMSE':en,'improvement_percent':100*(1-en/eb)})
    regional=pd.DataFrame(rows);regional.to_csv(OUT/'final_fidelity/regions.csv',index=False)
    # Reference checks are diagnostics only, never used to revise model selection.
    indices=np.unique(np.r_[np.argsort(abs(pn-y))[-8:],np.random.default_rng(206199).choice(len(y),16,replace=False)])
    checks=[]
    for i in indices:
        value,_=adaptive(a['params'][i],float(x[i]),float(t[i]))
        checks.append({'row':int(i),'gap':float(abs(value-y[i]))})
    gap=max(q['gap'] for q in checks)
    if gap>1e-8:raise AssertionError('Independent teacher audit failed; report cannot claim validated fidelity')
    save(OUT/'final_fidelity/independent_teacher_audit.json',{'checks':checks,'max_gap':gap})
    report_main();diagram(selected)
    gain=lambda k:100*(1-best[k]/base[k])
    pargain=100*(dev.loc[selected,'parameters_per_seed']/dev.loc['BASELINE','parameters_per_seed']-1)
    latency=100*(dev.loc[selected,'inference_1024_ms_per_seed']/dev.loc['BASELINE','inference_1024_ms_per_seed']-1)
    summary=f'''\n**Measured outcome: better global price fidelity, not a uniform accuracy-and-efficiency improvement.**

The selected published-modified-MLP correction reduced fresh price RMSE from
{base.price_RMSE:.8g} to {best.price_RMSE:.8g} (**{gain('price_RMSE'):.2f}%**), IV RMSE by
{gain('IV_RMSE_volatility_points'):.2f}%, P95 price error by {gain('price_P95'):.2f}% and maximum
price error by {gain('price_max'):.2f}%. All four unchanged fidelity gates pass.
It adds **{pargain:.2f}% parameters**, and measured batch inference latency changes
by **{latency:+.2f}%**. It is therefore not established as a more efficient solver.

There are material qualifications: development 7–30-day price RMSE worsened by
4.16%; fresh PDE RMSE worsened by {-gain('PDE_RMSE'):.2f}% and maximum PDE residual by
{-gain('PDE_max'):.2f}%, although PDE P95 improved by {gain('PDE_P95'):.2f}%.
The existing baseline remains unchanged and is **not replaced as the default**.
The selected checkpoint is a research candidate under the predeclared aggregate
gates, not proof of improvement in every difficult region or metric.

Modified gating reduced development price RMSE by 4.35% versus the similarly
sized plain correction. Gradient balancing improved IV error but worsened price
and full-domain PDE error. RAD did not recover the lost price accuracy. Neither
addition was retained. Candidate E was not triggered because A passed and improved;
D was already exact; Fourier features and gPINN were not justified.

**Verification:** 23 tests passed, one optional MLX test skipped. Ten trained runs
preserve their required parent tensors. Six RAD pools retain unique selected points
inside the original domain. Fresh fidelity inputs are disjoint from training,
development, collocation and RAD pools. {len(checks)} independent adaptive-integral
teacher checks agree within {gap:.3g}. No market quotes were parsed or scored.

'''
    path=ROOT/'PINN_ARCHITECTURE_IMPROVEMENT_REPORT.md';text=path.read_text();first,rest=text.split('\n',1)
    addendum=f'''\n## Reporting-only final checks

Common-IV comparison uses the same {common['quotes']} valid quotes in both arms:
baseline {common['baseline_IV_RMSE_vol_points']:.8g}, selected
{common['selected_IV_RMSE_vol_points']:.8g} volatility points. This prevents a
change in inversion coverage from being mistaken for an accuracy gain.

Fresh regional diagnostics (not used to revise selection):

{md(regional)}

The fixed published-state curve grid has zero detected monotonicity/convexity
violations in all tested arms, but this is not a domain-wide guarantee. On the
fresh PDE diagnostic states, the selected model still has mean per-seed convexity
violation rate {best.convexity_violation_percent:.6g}% and monotonicity violation
rate {best.monotonicity_violation_percent:.6g}%, at the separately documented tolerances.

Run the reporting-only `deliver` module after `report` to reproduce this summary,
common-IV/region checks and the parallel-branch architecture schematic. It does
not change the frozen selection, training code, metric thresholds or checkpoints.
'''
    path.write_text(first+'\n'+summary+rest+addendum)
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[path,ROOT/'CURRENT_PINN_BASELINE.md',ROOT/'PINN_ARCHITECTURE_ABLATION.csv',
        ROOT/'src/mentor_dh_pinn/research_modified_pinn.py',ROOT/'tests/test_research_modified_pinn.py']
    sources+=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json')
    save(OUT/'delivery_manifest.json',{'utc':old.stamp(),'reporting_only_addendum':True,'files':{str(p.relative_to(ROOT)):sha(p) for p in sources}})
    print('DELIVERED',path,'fresh price improvement',gain('price_RMSE'))


if __name__=='__main__':
    torch.set_num_threads(1);main()
