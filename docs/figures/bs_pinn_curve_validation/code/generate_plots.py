"""Visualization only. Load frozen v4 C3 weights; never train, fit or open market data.

Run with the repository's existing Python environment. All new artifacts stay
inside this directory. Existing experiment files are read-only and hashed.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[2]
EXP=ROOT/'experiments/nifty_multifactor_v4'
sys.path.insert(0,str(EXP));sys.path.insert(0,str(ROOT))
import run as existing
from literature_exact import black, exact, adaptive, admissible

FIG=HERE/'figures';DATA=HERE/'data'
COLORS={'C_BS':'#2563a6','C_exact_DH':'#db8b2b','C_PINN':'#087f73'}
LABELS={'C_BS':'Black-Scholes analytical','C_exact_DH':'Exact Double Heston','C_PINN':'Current PINN'}
STYLES={'C_BS':('-',1.9),'C_exact_DH':('-',2.8),'C_PINN':('--',1.8)}
CAPTION='K=100 illustrative price units. Exact DH is the PINN target; Black-Scholes is the shape reference.'
FIGURES=[]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def table(frame):
    def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def savefig(fig,name,caption=CAPTION):
    fig.text(.5,.025,caption,ha='center',va='bottom',fontsize=9,color='#455466',wrap=True)
    fig.savefig(FIG/name,dpi=300,facecolor='white')
    plt.close(fig);FIGURES.append(name)


def axis(ax,xlabel,title):
    ax.set(xlabel=xlabel,ylabel='European Call Option Price C',title=title)
    ax.grid(alpha=.18);ax.set_ylim(bottom=0)


def lines(ax,a,x,models):
    for model in models:
        ls,lw=STYLES[model];ax.plot(a[x],a[model],ls=ls,lw=lw,color=COLORS[model],label=LABELS[model])


def one_curve(a,x,models,name,title,cfg):
    fig,ax=plt.subplots(figsize=(9,5.8));fig.subplots_adjust(bottom=.19,top=.78,left=.12,right=.97)
    lines(ax,a,x,models);axis(ax,'Stock Price S' if x=='S' else 'Time to Maturity τ (years)','')
    fig.suptitle(title,y=.965,fontsize=14)
    ax.set_xlim(a[x].min(),a[x].max())
    if x=='S':
        ax.axvline(cfg['K'],ls=':',lw=1,color='#666666',label='S = K')
        detail=f"τ = {cfg['fixed_tau_years']*365:g} days; K = 100; r = q = 0"
    else:detail='S = K = 100; r = q = 0; all variance inputs fixed'
    fig.text(.5,.835,detail+'; BS σ = √(v_s + v_f) = 15.969%',ha='center',fontsize=10)
    ax.legend(frameon=False,fontsize=10,loc='upper left')
    savefig(fig,name)
    a.to_csv(DATA/name.replace('.png','.csv'),index=False)


def error_curve(a,x,name,title):
    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True);fig.subplots_adjust(bottom=.15,top=.89,hspace=.28,left=.14,right=.97)
    for ax,col,desc in zip(axs,['PINN_minus_exact_DH','abs_PINN_minus_exact_DH'],['Signed error','Absolute error']):
        ax.plot(a[x],a[col],color=COLORS['C_PINN'],lw=1.7);ax.set_ylabel(desc+' (price units)');ax.grid(alpha=.18)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axs[0].axhline(0,color='#777777',lw=.8);axs[1].set_ylim(bottom=0)
    axs[1].set_xlabel('Stock Price S' if x=='S' else 'Time to Maturity τ (years)')
    fig.suptitle(title,fontsize=14)
    savefig(fig,name,'PINN approximation error = current PINN − exact Double Heston. No Black-Scholes difference is used here.')
    a.to_csv(DATA/name.replace('.png','.csv'),index=False)


def surface_axis(ax,S,T,Z,cfg,zmax,title,error=False):
    ax.plot_surface(S,T,Z,cmap='magma' if error else 'viridis',vmin=0,vmax=zmax,
        rcount=S.shape[0],ccount=S.shape[1],linewidth=0,antialiased=True)
    ax.set(xlabel='Stock Price S',ylabel='Time to Maturity τ (years)',
        zlabel='Absolute price error' if error else 'Call Option Price C',title=title,
        xlim=cfg['S_range'],ylim=cfg['tau_range_years'],zlim=(0,zmax))
    ax.view_init(elev=26,azim=-125);ax.set_box_aspect((1.1,1,.85))
    ax.xaxis.labelpad=10;ax.yaxis.labelpad=12;ax.zaxis.labelpad=10


def shape_checks(a,variable,tolerance):
    h=float(np.diff(a[variable])[0]);rows=[];xx=a[variable].to_numpy()
    values={variable:xx,'first_difference_coordinate':np.pad((xx[:-1]+xx[1:])/2,(0,1),constant_values=np.nan),
        'second_difference_center_coordinate':np.pad(xx[1:-1],(0,2),constant_values=np.nan)}
    for model in LABELS:
        y=a[model].to_numpy();first=np.diff(y)/h;second=np.diff(y,2)/h**2
        for order,z in [('first',first),('second',second)]:
            values[model+'_'+order+'_difference']=np.pad(z,(0,len(y)-len(z)),constant_values=np.nan)
            rows.append({'model':model,'variable':variable,'derivative':order,'tested_locations':len(z),
                'minimum':float(z.min()),'negative_beyond_tolerance':int((z < -tolerance).sum()),
                'violation_percent':float(100*np.mean(z < -tolerance)),'tolerance':tolerance})
    return rows,pd.DataFrame(values)


def main():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11,'legend.fontsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','savefig.dpi':300})
    FIG.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)
    c=existing.verify();selection=read(EXP/'artifacts/pinn_selection.json');lock=read(EXP/'artifacts/evaluation_lock.json')
    assert selection['chosen']=='C3' and selection['width']==256 and selection['depth']==5
    assert read(EXP/'artifacts/controlled_results.json')['fidelity']['pass']
    models=existing.nets(c)
    for seed,net in zip(c['pinn']['seeds'],models):
        assert net.width==256 and net.depth==5 and net.factors==2
        assert selection['checkpoint_sha256'][str(seed)]==lock['models'][str(seed)]==digest(EXP/f'artifacts/pinn_s{seed}/weights.pt')
        net.requires_grad_(False)
    sources=[EXP/'config.json',EXP/'run.py',EXP/'literature_exact.py',ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',
        ROOT/'src/double_heston_reference.py',ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',
        EXP/'artifacts/manifest.json',EXP/'artifacts/pinn_selection.json',EXP/'artifacts/evaluation_lock.json',
        EXP/'artifacts/controlled_results.json',EXP/'artifacts/controlled_metrics.csv',EXP/'artifacts/controlled_predictions.csv',
        EXP/'artifacts/fidelity_predictions.csv',EXP/'artifacts/controlled_cases.json']
    sources += [EXP/f'artifacts/pinn_s{s}/weights.pt' for s in c['pinn']['seeds']]
    for name,expected in lock['baselines'].items():
        source=EXP/'artifacts/baselines'/name
        assert digest(source)==expected;sources.append(source)
    before={str(p.relative_to(ROOT)):digest(p) for p in sources}
    base=next(v for v in read(EXP/'artifacts/controlled_cases.json') if v['id']=='BASE_representative_0')
    p=np.array(base['params']);assert np.array_equal(p,c['published_double_slow_first']);admissible(p)
    cfg={'network_type':'Double-Heston PINN','experiment':'nifty_multifactor_v4','candidate':'C3',
        'checkpoint_paths':[str(EXP/f'artifacts/pinn_s{s}/weights.pt') for s in c['pinn']['seeds']],
        'checkpoint_sha256':selection['checkpoint_sha256'],'ensemble':'arithmetic mean of the two frozen price predictions, seeds 17 and 43',
        'architecture':'TorchRegularVariancePINN, 24 engineered features, five hidden layers x 256 tanh units, 269825 weights/biases per seed',
        'K':100.,'r':0.,'q':0.,'sigma_BS':float(np.sqrt(p[4]+p[9])),
        'sigma_rule':'Constant sqrt(v_slow + v_fast) from the existing BASE anchor, not a fitted per-option IV',
        'v1_slow':float(p[4]),'v2_fast':float(p[9]),'params_order':['kappa_s','theta_s','sigma_s','rho_s','v_s0','kappa_f','theta_f','sigma_f','rho_f','v_f0'],
        'params':p.tolist(),'scenario':'BASE_representative_0 unchanged',
        'S_range':(100*np.exp(np.array([-.36,.36]))).tolist(),'tau_range_years':[7/365,2.],
        'fixed_S':100.,'fixed_tau_years':90/365,'curve_points':401,'surface_shape':[121,121],
        'multiple_maturity_days':[30,90,180,365],'multiple_S_over_K':[.8,1.,1.2],
        'camera_elevation':26,'camera_azimuth':-125,'DPI':300,
        'units':'Illustrative price units with K=100; not observed NSE prices and not normalized neural outputs',
        'input':'coords=[ln(F/K),v_slow,v_fast,tau_years]; structural shape [n,2,4]=kappa,theta,sigma,rho slow then fast',
        'normalization':'F=S*exp((r-q)*tau), D=exp(-r*tau); net.price gives C/(D*K); existing.neural gives C/(D*F); displayed C=existing.neural*D*F',
        'output_transform':'IV=sqrt(expected_average_variance)*exp(1.8*tanh(head)); total variance=tau*IV^2; analytic Black call ansatz; exact terminal branch at tau=0',
        'training_domain':{'x':[-.36,.36],'tau_years':[7/365,2.],'v_slow':c['pinn']['slow_state_domain'],'v_fast':c['pinn']['fast_state_domain'],'scale':c['pinn']['scale_domain']},
        'shape_derivative_tolerance':1e-7,'price_bound_tolerance':1e-7,
        'terminal_check':'tau=0 checks hard-coded terminal construction; positive tau<7/365 is explicitly labelled extrapolation, not accepted-domain fidelity',
        'selection_policy':'Use already selected C3 and existing BASE; no choice made based on new plot scores',
        'existing_sources_sha256':before,'generator_sha256':digest(__file__),'market_files_opened':False,
        'metric_source':'Existing controlled_metrics.csv only: all 40 independent surfaces, their already scored held-out cells; no new final evaluation'}
    dump(HERE/'PLOT_CONFIGURATION.json',cfg)
    S=np.linspace(*cfg['S_range'],cfg['curve_points']);t=np.linspace(*cfg['tau_range_years'],cfg['curve_points'])
    fallbacks=[]
    def prices(spots,taus,terminal=False):
        spots,taus=np.broadcast_arrays(np.asarray(spots,float),np.asarray(taus,float));spots=spots.ravel();taus=taus.ravel()
        F=spots*np.exp((cfg['r']-cfg['q'])*taus);D=np.exp(-cfg['r']*taus);x=np.log(F/cfg['K'])
        assert np.all(abs(x)<=.36+1e-12)
        if not terminal:assert np.all(taus>=7/365-1e-14) and np.all(taus<=2+1e-14)
        bs=np.maximum(spots-cfg['K'],0.)
        positive=taus>0;bs[positive]=D[positive]*F[positive]*black(x[positive],taus[positive],cfg['sigma_BS'])
        neural,seeds=existing.neural(models,p,x,taus)
        if terminal:ref=np.array([adaptive(p,float(xx),float(tt))[0] for xx,tt in zip(x,taus)])
        else:ref=exact(p,x,taus,fallbacks)
        frame=pd.DataFrame({'S':spots,'tau_years':taus,'tau_days':taus*365,'K':cfg['K'],'F':F,'D':D,'x_log_F_over_K':x,
            'C_BS':bs,'C_exact_DH':ref*D*F,'C_PINN':neural*D*F,'C_PINN_s17':seeds[0]*D*F,'C_PINN_s43':seeds[1]*D*F})
        frame['PINN_minus_exact_DH']=frame.C_PINN-frame.C_exact_DH
        frame['abs_PINN_minus_exact_DH']=abs(frame.PINN_minus_exact_DH)
        frame['PINN_minus_BS_model_comparison']=frame.C_PINN-frame.C_BS
        frame['exact_DH_minus_BS_model_difference']=frame.C_exact_DH-frame.C_BS
        assert np.isfinite(frame.to_numpy()).all()
        assert np.allclose(frame.PINN_minus_BS_model_comparison,frame.PINN_minus_exact_DH+frame.exact_DH_minus_BS_model_difference,atol=1e-12)
        return frame
    # Independent formula check of spot/forward conversion, including nonzero carry.
    ss=np.array([80.,100.,120.]);tt=np.array([.1,.5,2.]);rr=.03;qq=.01;sig=cfg['sigma_BS'];ff=ss*np.exp((rr-qq)*tt)
    dd=(np.log(ss/100)+(rr-qq+sig**2/2)*tt)/(sig*np.sqrt(tt))
    spot_bs=ss*np.exp(-qq*tt)*ndtr(dd)-100*np.exp(-rr*tt)*ndtr(dd-sig*np.sqrt(tt))
    assert np.allclose(spot_bs,np.exp(-rr*tt)*ff*black(np.log(ff/100),tt,sig),atol=1e-12,rtol=1e-12)
    # Check current inference against existing recorded predictions without scoring any unopened data.
    archived=pd.read_csv(EXP/'artifacts/controlled_predictions.csv');arch=archived[archived.case.eq(base['id'])].iloc[::101]
    replay,_=existing.neural(models,p,arch.x.to_numpy(),arch.tau.to_numpy())
    assert np.allclose(replay,arch.DH_PINN.to_numpy(),atol=5e-14,rtol=1e-11)
    cs=prices(S,cfg['fixed_tau_years']);ct=prices(cfg['fixed_S'],t)
    for number,mods in [(1,['C_BS']),(2,['C_PINN']),(3,list(LABELS))]:
        names={1:'01_black_scholes_C_vs_S.png',2:'02_pinn_C_vs_S.png',3:'03_C_vs_S_BS_vs_PINN.png'}
        titles={1:'Black-Scholes: Option Price vs Stock Price',2:'Current Double Heston PINN: Option Price vs Stock Price',3:'Option Price vs Stock Price: Black-Scholes and PINN'}
        one_curve(cs,'S',mods,names[number],titles[number],cfg)
    error_curve(cs,'S','04_C_vs_S_error.png','Stock-price slice: PINN error against exact Double Heston')
    for number,mods in [(5,['C_BS']),(6,['C_PINN']),(7,list(LABELS))]:
        names={5:'05_black_scholes_C_vs_tau.png',6:'06_pinn_C_vs_tau.png',7:'07_C_vs_tau_BS_vs_PINN.png'}
        titles={5:'Black-Scholes: Option Price vs Time to Maturity',6:'Current Double Heston PINN: Option Price vs Time to Maturity',7:'Option Price vs Time to Maturity:\nBlack-Scholes, Exact Model and PINN'}
        one_curve(ct,'tau_years',mods,names[number],titles[number],cfg)
    error_curve(ct,'tau_years','08_C_vs_tau_error.png','Maturity slice: PINN error against exact Double Heston')
    print('1D curves generated',flush=True)
    sg=np.linspace(*cfg['S_range'],121);tg=np.linspace(*cfg['tau_range_years'],121);SS,TT=np.meshgrid(sg,tg)
    mesh=prices(SS,TT);mesh.to_csv(DATA/'surface_all_models.csv',index=False)
    zmax=1.04*float(mesh[list(LABELS)].max().max());cfg['shared_surface_C_limits']=[0,zmax]
    surface_names=['09_black_scholes_C_S_tau_3D.png','11_exact_DH_C_S_tau_3D.png','10_pinn_C_S_tau_3D.png']
    for name,model_name in zip(surface_names,LABELS):
        fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(projection='3d');fig.subplots_adjust(bottom=.15,top=.88,left=.02,right=.94)
        surface_axis(ax,SS,TT,mesh[model_name].to_numpy().reshape(SS.shape),cfg,zmax,LABELS[model_name])
        savefig(fig,name);mesh[['S','tau_years','K','F','D',model_name]].to_csv(DATA/name.replace('.png','.csv'),index=False)
    fig=plt.figure(figsize=(18,6));fig.subplots_adjust(bottom=.19,top=.85,left=.015,right=.97,wspace=.1)
    for i,model_name in enumerate(LABELS):surface_axis(fig.add_subplot(1,3,i+1,projection='3d'),SS,TT,mesh[model_name].to_numpy().reshape(SS.shape),cfg,zmax,LABELS[model_name])
    fig.suptitle('Identical grids, price limits and viewing angles',fontsize=16)
    savefig(fig,'11b_BS_exact_DH_PINN_3D_comparison.png')
    ee=mesh.abs_PINN_minus_exact_DH.to_numpy().reshape(SS.shape)
    fig=plt.figure(figsize=(9,7));fig.subplots_adjust(bottom=.15,top=.9,right=.92);ax=fig.add_subplot(projection='3d')
    surface_axis(ax,SS,TT,ee,cfg,float(ee.max())*1.02,'PINN absolute error vs exact Double Heston',error=True)
    savefig(fig,'12_pinn_absolute_error_3D.png','Absolute numerical error, in the same illustrative K=100 price units; no error clipping.')
    fig,ax=plt.subplots(figsize=(9,6));fig.subplots_adjust(bottom=.18,top=.88,left=.11,right=.90)
    im=ax.pcolormesh(SS,TT,ee,cmap='magma',shading='nearest',vmin=0,vmax=ee.max())
    ax.set(xlabel='Stock Price S',ylabel='Time to Maturity τ (years)',title='PINN absolute error vs exact Double Heston')
    fig.colorbar(im,ax=ax,label='Absolute price error');savefig(fig,'12_pinn_absolute_error_heatmap.png','Full shared plotting grid retained, including the largest deviations.')
    mesh.to_csv(DATA/'12_pinn_absolute_error_grid.csv',index=False)
    for number,ref,name in [(13,'C_BS','Black-Scholes'),(14,'C_exact_DH','exact teacher')]:
        fig,ax=plt.subplots(figsize=(7,7));fig.subplots_adjust(bottom=.18,top=.87,left=.13,right=.95)
        ax.scatter(mesh[ref],mesh.C_PINN,s=3,alpha=.18,c=COLORS['C_PINN'],rasterized=True,linewidths=0,label='All 14,641 matched grid pairs')
        lim=1.03*max(mesh[ref].max(),mesh.C_PINN.max());ax.plot([0,lim],[0,lim],'--',c='#c3374b',lw=1.1,label='y = x (reference, not fitted)')
        ax.set(xlabel='Black-Scholes analytical price' if number==13 else 'Exact Double Heston price',ylabel='PINN predicted price',xlim=(0,lim),ylim=(0,lim),
            title='PINN Option Price vs Black-Scholes Analytical Price' if number==13 else 'PINN Option Price vs Exact Double Heston Price')
        ax.set_aspect('equal',adjustable='box');ax.legend(loc='upper left',fontsize=9);ax.grid(alpha=.15)
        savefig(fig,'13_PINN_vs_BlackScholes_parity.png' if number==13 else '14_PINN_vs_exact_teacher_parity.png',
            'BS comparison is a model-difference diagnostic; equality is not required.' if number==13 else 'This is numerical fidelity to the PINN’s actual target. Error magnitudes are in Figure 12.')
    mesh.to_csv(DATA/'13_14_parity_matched_pairs.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(12,9));fig.subplots_adjust(bottom=.12,top=.88,hspace=.36,wspace=.25);slices=[]
    for ax,day in zip(axs.ravel(),cfg['multiple_maturity_days']):
        a=prices(S,day/365);slices.append(a);lines(ax,a,'S',list(LABELS));axis(ax,'Stock Price S',f'τ = {day} days')
        ax.axvline(100,color='#777777',ls=':',lw=.8);ax.set_xlim(cfg['S_range']);ax.legend(fontsize=8,loc='upper left',frameon=False)
    fig.suptitle('Option price vs stock price across maturities — unchanged BASE parameters',fontsize=15)
    savefig(fig,'15_multiple_maturity_C_vs_S.png');pd.concat(slices).to_csv(DATA/'15_multiple_maturity_C_vs_S.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(15,5.7));fig.subplots_adjust(bottom=.21,top=.81,wspace=.3,left=.06,right=.98);time_slices=[]
    for ax,ratio in zip(axs,cfg['multiple_S_over_K']):
        a=prices(100*ratio,t);time_slices.append(a);lines(ax,a,'tau_years',list(LABELS));axis(ax,'Time to Maturity τ (years)',f'S/K = {ratio:g} ({"OTM" if ratio<1 else "ITM" if ratio>1 else "ATM"})')
        ax.set_xlim(cfg['tau_range_years']);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Maturity curves at fixed OTM, ATM and ITM stock prices',fontsize=15)
    savefig(fig,'16_multiple_moneyness_C_vs_tau.png');pd.concat(time_slices).to_csv(DATA/'16_multiple_moneyness_C_vs_tau.csv',index=False)
    print('Surfaces and multiple slices generated',flush=True)
    old=pd.read_csv(EXP/'artifacts/controlled_metrics.csv');old=old[old.kind.eq('independent') & old.model.isin(['BS_SELECTED','SH_BEST_FOUND','DH_PINN'])]
    assert len(old)==120 and old.case.nunique()==40
    old.to_csv(DATA/'17_existing_per_surface_metrics.csv',index=False)
    stats=old.groupby('model').agg(mean_RMSE=('price_RMSE','mean'),mean_MAE=('price_MAE','mean'),mean_P95=('price_P95','mean'),
        worst_max=('price_max','max'),mean_IV_RMSE=('IV_RMSE_volatility_points','mean'),scored_quotes=('quotes','sum'),valid_IV_quotes=('IV_valid_quotes','sum')).reindex(['BS_SELECTED','SH_BEST_FOUND','DH_PINN'])
    stats.to_csv(DATA/'17_existing_metric_summary.csv')
    fig,axs=plt.subplots(2,3,figsize=(15,8));fig.subplots_adjust(bottom=.14,top=.84,wspace=.36,hspace=.42)
    display=['BS selected\n(flat/term)','Best-fit\nSingle Heston','Current\nDH-PINN']
    for ax,key,title in zip(axs.ravel(),stats.columns[:5],['Mean surface RMSE','Mean surface MAE','Mean surface P95 absolute error','Worst absolute error (all surfaces)','Mean surface IV RMSE']):
        vals=stats[key].to_numpy();ax.barh(display,vals,color=['#2563a6','#8581bc','#087f73']);ax.invert_yaxis();ax.set_xlim(0,vals.max()*1.35)
        for i,v in enumerate(vals):ax.text(v+vals.max()*.035,i,f'{v:.3g}',va='center',fontsize=10)
        ax.set_title(title);ax.set_xlabel('Volatility percentage points' if key=='mean_IV_RMSE' else 'Forward-normalized price C/(D·F)');ax.grid(axis='x',alpha=.15)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}'));ax.xaxis.set_major_locator(MaxNLocator(3))
    axs.ravel()[5].axis('off');axs.ravel()[5].text(0,.95,'Existing evaluated data only\n\n40 independent surfaces\nSame held-out cells for all prices\nNo recalibration or new test opening\n\nExact DH generated these targets;\nits tautological zero is not a bar.\n\nIV-valid counts are in the CSV;\nthey differ by model.',va='top',fontsize=11)
    fig.suptitle('Existing controlled benchmark — not the new BASE visualization grid',fontsize=15)
    savefig(fig,'17_existing_evaluation_metrics.png','BS uses its already selected calibration-only term/flat baseline here, not the constant-vol shape reference in Figures 1–16.')
    errors=mesh.abs_PINN_minus_exact_DH.to_numpy();fig,axs=plt.subplots(1,2,figsize=(12,5.6));fig.subplots_adjust(bottom=.2,top=.84,left=.08,right=.97,wspace=.28)
    counts,edges=np.histogram(errors,bins=60);axs[0].hist(errors,bins=edges,color=COLORS['C_PINN'],alpha=.85)
    axs[0].set(xlabel='Absolute price error',ylabel='Grid-point count',title='Full error distribution (no tail removal)')
    ordered=np.sort(errors);prob=np.arange(1,len(ordered)+1)/len(ordered);axs[1].plot(ordered,prob,color=COLORS['C_PINN'])
    axs[1].set(xlabel='Absolute price error',ylabel='Cumulative fraction',title='Empirical cumulative distribution',ylim=(0,1.01),xlim=(0,ordered.max()*1.03))
    for ax in axs:ax.grid(alpha=.18);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
    fig.suptitle('PINN vs exact Double Heston: all 14,641 illustrative grid points',fontsize=14)
    savefig(fig,'18_PINN_error_distribution.png','Error units correspond to K=100. This descriptive grid is not a new independent final test.')
    pd.DataFrame({'bin_left':edges[:-1],'bin_right':edges[1:],'count':counts}).to_csv(DATA/'18_error_histogram.csv',index=False)
    pd.DataFrame({'absolute_error':ordered,'ECDF':prob}).to_csv(DATA/'18_error_ECDF.csv',index=False)
    # Numerical curve checks, not repairs. Derivative stencils are explicitly indexed.
    tol=cfg['shape_derivative_tolerance'];checks=[]
    for label,a,xvar in [('S_slice_90d',cs,'S'),('tau_slice_ATM',ct,'tau_years')]:
        rows,df=shape_checks(a,xvar,tol)
        for row in rows:row['sample']=label;row['required_sign']=row['derivative']=='first' or xvar=='S'
        checks+=rows;df.to_csv(DATA/f'{label}_finite_differences.csv',index=False)
    for model_name in LABELS:
        z=mesh[model_name].to_numpy().reshape(SS.shape);h=sg[1]-sg[0]
        for order,dd in [('first',np.diff(z,axis=1)/h),('second',np.diff(z,n=2,axis=1)/h**2)]:
            checks.append({'sample':'whole_surface_S_direction','model':model_name,'variable':'S','derivative':order,'tested_locations':dd.size,
                'minimum':float(dd.min()),'negative_beyond_tolerance':int((dd < -tol).sum()),'violation_percent':float(100*np.mean(dd < -tol)),'tolerance':tol,'required_sign':True})
    checkframe=pd.DataFrame(checks);checkframe.to_csv(DATA/'curve_shape_checks.csv',index=False)
    # Endpoint is explicitly implemented by the existing ansatz. Positive sub-7-day values are diagnostic extrapolation only.
    tiny=np.array([0.,1e-7,1e-6,1e-5,1e-4,1e-3,7/365]);term=[]
    for ratio in cfg['multiple_S_over_K']:
        a=prices(100*ratio,tiny,terminal=True);a['payoff']=max(100*ratio-100,0);a['outside_positive_training_tau']=(a.tau_years>0)&(a.tau_years<7/365)
        for model_name in LABELS:a[model_name+'_minus_payoff']=a[model_name]-a.payoff
        term.append(a)
    terminal=pd.concat(term);terminal.to_csv(DATA/'19_terminal_diagnostic.csv',index=False)
    terminal_zero=terminal[terminal.tau_years.eq(0)]
    assert max(abs(terminal_zero[k]-terminal_zero.payoff).max() for k in LABELS)<1e-11
    fig,axs=plt.subplots(1,3,figsize=(15,5.8));fig.subplots_adjust(bottom=.24,top=.83,wspace=.3,left=.06,right=.98)
    for ax,(ratio,a) in zip(axs,zip(cfg['multiple_S_over_K'],term)):
        b=a[a.tau_years>0]
        for model_name in LABELS:ax.plot(b.tau_years,b[model_name+'_minus_payoff'],label=LABELS[model_name],color=COLORS[model_name],ls=STYLES[model_name][0],marker='.',ms=4)
        ax.axhline(0,lw=.8,color='#888888');ax.set(xscale='log',xlabel='Time to Maturity τ (years, log)',ylabel='Price minus terminal payoff',title=f'S/K = {ratio:g}')
        ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
    fig.suptitle('Terminal diagnostic — positive maturities below 7 days are extrapolation',fontsize=14)
    savefig(fig,'19_terminal_diagnostic.png','τ = 0 is exact by construction, not learned accuracy. Sub-7-day curves do not extend the validated training domain.')
    gridmetrics={'price_RMSE':float(np.sqrt(np.mean(errors**2))),'price_MAE':float(errors.mean()),'price_P95':float(np.quantile(errors,.95)),'price_max':float(errors.max()),
        'max_BS_DH_model_difference':float(abs(mesh.exact_DH_minus_BS_model_difference).max()),'grid_points':len(mesh)}
    bounds={}
    for model_name in LABELS:
        lower=np.maximum(mesh.S.to_numpy()-cfg['K'],0);y=mesh[model_name].to_numpy()
        bounds[model_name]={'below_intrinsic_beyond_tolerance':int((y<lower-cfg['price_bound_tolerance']).sum()),
            'above_spot_beyond_tolerance':int((y>mesh.S.to_numpy()+cfg['price_bound_tolerance']).sum()),'minimum_time_value':float((y-lower).min())}
    dump(DATA/'numerical_checks.json',{'config':cfg,'grid_metrics':gridmetrics,'shape_checks':checks,'price_bounds':bounds,
        'terminal_zero_max_error':float(max(abs(terminal_zero[k]-terminal_zero.payoff).max() for k in LABELS)),
        'spot_forward_BS_conversion_pass':True,'checkpoint_replay_pass':True,'exact_adaptive_fallback_points':len(fallbacks),
        'no_price_clipping':True,'no_training_or_new_final_data':True})
    dump(DATA/'exact_reference_fallbacks.json',fallbacks)
    dump(HERE/'PLOT_CONFIGURATION.json',cfg)
    descriptions=[
        ('Black-Scholes C vs S','01_black_scholes_C_vs_S.png','401 stock prices; τ=90/365 years. This is the smooth constant-volatility shape reference, not a DH numerical target.'),
        ('Current PINN C vs S','02_pinn_C_vs_S.png','Identical S grid and fixed 90-day maturity. The neural curve is denormalized into K=100 price units. Check its shape violations below rather than assuming arbitrage freedom.'),
        ('Stock-price overlay','03_C_vs_S_BS_vs_PINN.png','All three curves use identical S and carry. Orange/teal separation is approximation error; blue/orange separation is a legitimate difference between the fixed-volatility and stochastic-volatility models.'),
        ('Stock-price error','04_C_vs_S_error.png','Signed and absolute PINN-minus-exact-DH errors at 90 days. These show deviations hidden by overlapping price curves; BS differences are not used as errors.'),
        ('Black-Scholes C vs τ','05_black_scholes_C_vs_tau.png','401 maturities, S=K=100. Volatility is one constant, not each point’s own implied volatility. With r=q=0 the call gains time value.'),
        ('Current PINN C vs τ','06_pinn_C_vs_tau.png','Identical maturities and fixed S, variance states and structural coefficients. This is a maturity sweep, not a forecast of future market dates.'),
        ('Maturity overlay','07_C_vs_tau_BS_vs_PINN.png','All three maturity curves share the exact grid. The DH expected variance changes with horizon, so departure from the constant-vol BS curve is not evidence of PINN failure.'),
        ('Maturity error','08_C_vs_tau_error.png','Signed/absolute PINN-minus-DH errors reveal maturity-localized approximation deviations at S=100. No model is refitted along the curve.'),
        ('Black-Scholes surface','09_black_scholes_C_S_tau_3D.png','121×121 genuine computed mesh. The full common spot/maturity/price limits and camera are used, not normalized outputs mislabeled as prices.'),
        ('Current PINN surface','10_pinn_C_S_tau_3D.png','The exact same mesh, color range, camera and axes as Figures 9 and 11. Shape resemblance to BS is appropriate; numerical equality to BS is not required.'),
        ('Exact DH surface and identical-axis comparison','11_exact_DH_C_S_tau_3D.png','Exact numerical Fourier pricing at the same parameters as the network. The additional side-by-side figure makes all three directly comparable, with the same price limits.'),
        ('Absolute error surface and heatmap','12_pinn_absolute_error_3D.png','Each value is |PINN−exact DH| in K=100 price units. The heatmap uses the same complete grid; maximum errors are neither clipped nor removed.'),
        ('PINN vs BS parity','13_PINN_vs_BlackScholes_parity.png','All 14,641 points pair the same (S,τ). The diagonal y=x is fixed, not a fitted regression. This is a model-difference diagnostic, not PINN fidelity.'),
        ('PINN vs exact DH parity','14_PINN_vs_exact_teacher_parity.png','Same paired grid with the correct numerical target. Nearly coincident points should be interpreted together with Figure 12 because the full price scale can obscure small errors.'),
        ('Multiple maturity stock-price slices','15_multiple_maturity_C_vs_S.png','30, 90, 180 and 365 days, each using the same 401 stock prices. Only maturity changes between panels; all parameters stay fixed.'),
        ('OTM/ATM/ITM maturity slices','16_multiple_moneyness_C_vs_tau.png','S/K=0.8, 1 and 1.2; all non-spot inputs fixed. All three models use exactly the same maturity values in each panel.'),
        ('Existing evaluation metrics','17_existing_evaluation_metrics.png','Only already saved scores: 40 independent controlled surfaces with 1,006 held-out cells each. RMSE/MAE/P95/IV bars average per-surface values; max is the worst across surfaces. These are normalized benchmark errors, not K=100 plot-grid scores. BS here is its previously selected fitted baseline, not the constant-vol shape comparator.'),
        ('Full error distribution','18_PINN_error_distribution.png','Histogram and ECDF retain every new diagnostic-grid absolute error. This is descriptive shape/fidelity evidence, not an independent final-test result.'),
        ('Terminal payoff diagnostic','19_terminal_diagnostic.png','At τ=0 the existing network explicitly enforces payoff. Positive τ below 7/365 lies outside training and is labelled extrapolation, not proof of learned short-expiry accuracy.')]
    checkpoint_text='v4 C3, arithmetic mean of seeds 17 and 43; exact paths and hashes in PLOT_CONFIGURATION.json.'
    param_text='K=100, r=q=0, BS σ=√0.0255='+f'{cfg["sigma_BS"]:.10f}'+', slow (κ,θ,σ,ρ,v₀)=(0.9491,0.0257,0.0517,0.7009,0.0003), fast=(10.7526,0.033,0.3613,−0.8916,0.0252).'
    checks_required=checkframe[checkframe.required_sign]
    report=['# Current Double-Heston PINN — option-price curve visualization',
        '**Network category: Double-Heston PINN, not a Black-Scholes PINN.** This is visualization only; no retraining, tuning, changed splits, changed model parameters or opening of untouched final data.',
        '## Model recovered from the repository',checkpoint_text,
        'The previously selected C3 is 5 hidden tanh layers ×256 units, 24 engineered inputs and 269,825 parameters per seed. Existing selection/lock/checkpoint hashes match. C1/C2 are not silently substituted. v3 was rejected on development and is not the displayed model. All existing experiment source hashes were verified.',
        '## Price convention and common configuration',param_text,
        'The strike of 100 is an illustrative unit scale, not a historical NSE strike or observed price. No market quotes are used in these new curves. The BASE representative parameter vector is unchanged. BS matches the initial total volatility only; its constant σ does not make it mathematically equivalent to stochastic volatility.',
        'F=S exp((r−q)τ), D=exp(−rτ), x=ln(F/K), τ in years. Here r=q=0, hence F=S and D=1. Coordinates are [x,v_slow,v_fast,τ], plus structural [κ,θ,σ,ρ] for each factor. `net.price` returns C/(D K), whereas the reused `run.neural` returns C/(D F). **All new price curves multiply that output by D F.** Calling `net.forward` directly would return IV, not price; we do not do that.',
        'The input features use x/0.36, volatility-scaled moneyness, log-scaled τ and variance/parameter transforms. The network learns a bounded log-IV correction to expected average variance: IV=√v̄·exp(1.8 tanh(head)); total implied variance=τ·IV², then the analytic Black call map produces price. This construction ensures payoff at zero maturity but does not alone guarantee global convexity. The slow/fast ordering is preserved.',
        f'S range [{cfg["S_range"][0]:.6f}, {cfg["S_range"][1]:.6f}] corresponds to x∈[−0.36,0.36]. This uses the valid model domain rather than extrapolating to S/K=0.5. Main τ range is 7/365–2 years. Curves use 401 points; surfaces 121×121; all PNGs are 300 DPI. Full 3D price range is [0, {zmax:.6f}].',
        '## What “follows” means here',
        '**Shape reference:** Black-Scholes. **Numerical target:** the exact same Double Heston model. PINN−DH is approximation error. Exact DH−BS is a model difference. PINN−BS combines both; it must not be called pure PINN error. These are diagnostic plots, not proof of universal accuracy, market fit or ten-parameter recovery.',
        '## Numerical results on the new illustrative grid',table(pd.DataFrame([gridmetrics])),
        'All units in this grid table are illustrative K=100 price units. These values do not replace the frozen evaluation metrics.',
        '## “Following the curve” checks',table(checks_required[['sample','model','derivative','tested_locations','negative_beyond_tolerance','violation_percent','minimum']]),
        'First differences use adjacent-pair secants; second differences use centered three-point stencils. The S grid is uniform. Derivative tolerance is 1e−7 in the corresponding units. Negative maturity second derivatives are not violations: maturity convexity is not required. No violations are repaired. Grid tests are finite-domain evidence, not a proof of global arbitrage freedom.',
        'The network and BS functions are analytically smooth for strictly positive maturity inside the domain. Full finite-difference values are saved, so local irregularities are inspectable. Maturity-monotonicity counts above apply to the controlled r=q=0 setup, not arbitrary dividend settings.',
        '## Price bounds and terminal condition',table(pd.DataFrame([{'model':k,**v} for k,v in bounds.items()])),
        'Bounds use tolerance 1e−7 price units. Reference roundoff is retained. At τ=0, all three prices agree with max(S−K,0) to within 1e−11 for S=80,100,120. For positive sub-7-day maturities, Figure 19 and its CSV report extrapolation explicitly. Endpoint agreement is hard-coded mathematical construction, not validation on short-dated data.',
        '## Figure-by-figure interpretation']
    def observed(a):
        e=a.PINN_minus_exact_DH.to_numpy()
        return f'Observed PINN–exact DH RMSE {np.sqrt(np.mean(e*e)):.6g}, maximum absolute deviation {abs(e).max():.6g} price units. This is approximation error; the separate BS–DH gap is a model difference.'
    slice_observations={}
    for number in range(1,20):
        if number in [1,5,9]:note='Analytical shape reference only; agreement with the PINN is not required. The monotonicity/convexity checks for BS show zero violations on the reported stock-price grids.'
        elif number<=4:note=observed(cs)+' The PINN increases and is convex in S at every tested stencil (zero violations at tolerance 1e−7).'
        elif number<=8:note=observed(ct)+' The PINN has zero detected decreasing-maturity intervals in this r=q=0 setup.'
        elif number==15:note=' '.join(f'{day} days: '+observed(a) for day,a in zip(cfg['multiple_maturity_days'],slices))
        elif number==16:note=' '.join(f'S/K={ratio:g}: '+observed(a) for ratio,a in zip(cfg['multiple_S_over_K'],time_slices))
        elif number==17:note='The saved controlled scores rank DH-PINN below recalibrated SH and selected BS on the displayed aggregates. That is a DH-generated synthetic benchmark result, not market superiority or equality with Black-Scholes.'
        elif number==19:note='All three endpoint payoffs match within 1e−11. Nonzero sub-7-day values are explicit extrapolation and are retained without repair; exact endpoint matching is built into the architecture.'
        else:note=observed(mesh)+' Over the surface, no tested S-monotonicity or S-convexity stencils violate the tolerance; finite-grid checks are not a global proof.'
        slice_observations[number]=note
    for number,(title,name,description) in enumerate(descriptions,1):
        specific=('Parameter sets and states vary across the 40 previously frozen surfaces; full teacher values are in data/17_existing_case_parameters.json and calibrated SH/BS values in data/17_existing_baselines.json. No new common BASE is asserted for this existing benchmark.' if number==17 else param_text)
        functions='Black-Scholes: existing literature_exact.black (forward-normalized analytical formula), converted with D·F. Exact reference: literature_exact.exact, with its unchanged adaptive Fourier fallback. PINN: run.nets/run.neural → TorchRegularVariancePINN.price.'
        if number==17:functions='Replots existing controlled_metrics.csv, generated by the frozen run.metrics/exact/bs_predict/neural workflow; no new scoring or calibration.'
        if number==19:functions='Existing literature_exact.adaptive for exact DH at tiny τ; run.neural and the model’s hard terminal branch; analytical BS with explicit τ=0 payoff.'
        report += [f'### {number}. {title}',description,f'**Parameters:** {specific}',f'**Checkpoint:** {checkpoint_text}',f'**Functions:** {functions}',f'**Interpretation:** {slice_observations[number]}',f'![{title}](figures/{name})']
        if number==11:report+=['![Identical-axis 3D comparison](figures/11b_BS_exact_DH_PINN_3D_comparison.png)']
        if number==12:report+=['![Absolute error heatmap](figures/12_pinn_absolute_error_heatmap.png)']
    report+=['## Provenance, source data and reproduction',
        'PLOT_CONFIGURATION.json contains parameters, checkpoint identities and source hashes. data/numerical_checks.json contains the numerical checks. All figure data are saved as CSVs; the shared surface/parity CSV contains every matched point and both seed predictions. DATA is illustrative model output except Figure 17, which explicitly reuses already evaluated controlled results. No observed option data are fabricated.',
        'Run `python code/generate_plots.py` from this directory using the existing project environment. It writes only this visualization directory. Existing frozen sources are verified before and after generation. The original scientific experiment, checkpoints and results remain untouched. The plotting script follows the Ponytail skill by reusing the existing model loader and exact/Black pricing functions instead of implementing a new model.',
        '## Limitations',
        'One pre-existing representative configuration is visualized, not all possible parameters. Farther tails outside the trained x range are not tested. Close-looking price curves can hide small numerical errors; residual plots disclose them. New dense grids are explicitly post-training diagnostics, not independent final-test evidence. The archive’s controlled results use a DH-generated target, not neutral market truth. Existing IV metrics have model-specific valid-inversion counts, included in the source CSV.']
    (HERE/'PLOT_REPORT.md').write_text('\n\n'.join(report)+'\n')
    dump(DATA/'17_existing_case_parameters.json',read(EXP/'artifacts/controlled_cases.json'))
    dump(DATA/'17_existing_baselines.json',{p.stem:read(p) for p in sorted((EXP/'artifacts/baselines').glob('*.json'))})
    existing.verify()
    assert before=={str(p.relative_to(ROOT)):digest(p) for p in sources},'Existing scientific artifact changed during visualization'
    images=[]
    for name in FIGURES:
        path=FIG/name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert min(image.size)>=1000 and min(image.info['dpi'])>=299
            images.append({'file':name,'width':image.width,'height':image.height,'DPI':image.info['dpi'],'sha256':digest(path)})
    dump(HERE/'OUTPUT_MANIFEST.json',{'generator_sha256':digest(__file__),'existing_source_hashes_unchanged':True,'figure_count':len(images),
        'images':images,'files':{str(p.relative_to(HERE)):digest(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='OUTPUT_MANIFEST.json' and '__pycache__' not in str(p)}})
    print(json.dumps({'figures':len(images),'grid_metrics':gridmetrics,'required_shape_violations':int(checks_required.negative_beyond_tolerance.sum()),'report':str(HERE/'PLOT_REPORT.md')},indent=2),flush=True)


if __name__=='__main__':main()
