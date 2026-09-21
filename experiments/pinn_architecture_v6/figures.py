"""Plots from saved logs only; no selection, training or market reads."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .common import *

SHORT={'BASELINE':'Baseline','A0_CONTINUE':'Continue','CONTROL_PLAIN':'Plain control',
       'ARCH_A_MODIFIED_MLP':'A: modified MLP','ARCH_B_MODIFIED_MLP_GRADBAL':'B: + balancing',
       'ARCH_C_MODIFIED_MLP_GRADBAL_RAD':'C: + RAD','ARCH_E_ADAPTIVE_ACTIVATION':'E: adaptive tanh'}


def training_plots():
    files=sorted(OUT.glob('*/seed_*/training.csv'))
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for path in files:
        arm=path.parent.parent.name;seed=int(path.parent.name.split('_')[-1]);a=pd.read_csv(path)
        for ax,col in zip(axes.flat,['total_loss','teacher','iv','pde','convexity','terminal_loss']):
            ax.plot(a.step,np.maximum(a[col],1e-25) if col!='terminal_loss' else a[col],
                    label=f'{SHORT[arm]} s{seed}',ls='-' if seed==17 else '--',alpha=.8)
            ax.set(title=col,xlabel='Adam step',ylabel='Raw loss' if col!='total_loss' else 'Weighted loss')
            if col!='terminal_loss':ax.set_yscale('log')
    axes[0,0].legend(fontsize=6,ncol=2);axes[1,2].set_title('Terminal: exact zero by inherited ansatz')
    fig.savefig(dest/'training_losses.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for path in sorted(OUT.glob('*/seed_*/gradients.csv')):
        arm=path.parent.parent.name;seed=int(path.parent.name.split('_')[-1]);a=pd.read_csv(path)
        if seed!=17:continue
        for name,g in a.groupby('component'):
            if name=='convexity':continue
            label=f'{SHORT[arm]} / {name}'
            axes[0].plot(g.step,g.weight,label=label)
            axes[1].plot(g.step,np.maximum(g.scaled_grad_l2*g.weight,1e-16),label=label)
    axes[0].set(yscale='log',title='Adaptive multipliers (seed 17)',xlabel='Adam step',ylabel='Weight; base normalizers unchanged')
    axes[1].set(yscale='log',title='Effective gradient norms (seed 17)',xlabel='Adam step',ylabel='L2 norm')
    axes[0].legend(fontsize=5,ncol=2);fig.savefig(dest/'adaptive_weights_and_gradients.png',dpi=160);plt.close(fig)
    # Collocation density: unchanged global support plus RAD-selected locations.
    original=data('collocation')['coords']
    for path in sorted(OUT.glob('ARCH_C*/seed_*/rad_1500.npz')):
        a=np.load(path);after=np.vstack([original[:-4096],a['selected']])
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,z,title in zip(axes,[original,after],['Initial 18,000 points','After RAD: 18,000 points']):
            im=ax.hist2d(z[:,0],np.log10(z[:,-1]*365),bins=(30,24),range=[[-.36,.36],[np.log10(7),np.log10(730)]],vmin=0,vmax=200,cmap='viridis')
            ax.set(xlabel='log(F/K)',ylabel='log10 maturity days',title=title)
        fig.colorbar(im[3],ax=axes,label='Count (color scale capped at 200; locations not filtered)')
        fig.savefig(dest/f'collocation_density_{path.parent.name}.png',dpi=150);plt.close(fig)


def metrics_plots():
    table=pd.read_csv(OUT/'PINN_ARCHITECTURE_ABLATION.csv');dest=OUT/'figures'
    labels=table.arm.map(SHORT)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,col,title in zip(axes.flat,['price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','parameters_per_seed','training_seconds_mean'],
          ['Development price RMSE','IV RMSE (vol points)','P95 price error','Max price error','Parameters per seed','Incremental training seconds']):
        ax.bar(labels,table[col],color='#3474ad');ax.tick_params(axis='x',rotation=40,labelsize=7);ax.set_title(title)
    axes[1,2].text(.02,.98,'Baseline: historical v5 increment only\nInherited training cost reported separately',transform=axes[1,2].transAxes,va='top',fontsize=7)
    fig.savefig(dest/'architecture_metrics.png',dpi=150);plt.close(fig)
    seeds=pd.read_csv(OUT/'per_seed_development.csv')
    stats=seeds.groupby('arm').agg({col:['mean','std','min','max'] for col in ['price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','PDE_RMSE']})
    stats.to_csv(OUT/'seed_robustness.csv')
    pre=pd.read_csv(OUT/'before_lbfgs_comparison.csv')
    post=seeds[seeds.arm.ne('BASELINE')]
    joined=pre.merge(post,on=['arm','seed'],suffixes=('_before','_after'))
    cols=['price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','PDE_RMSE']
    for col in cols:joined[col+'_improvement_percent']=100*(1-joined[col+'_after']/joined[col+'_before'])
    joined.to_csv(OUT/'lbfgs_effect.csv',index=False)


def diagram(selected):
    dest=OUT/'figures';fig,axes=plt.subplots(2,1,figsize=(13,6),layout='constrained')
    rows=[['log(F/K), τ, v_s, v_f\nconditional structural parameters','24 engineered features\n5 × 256 tanh core','v5: 3 × width-32\nshared correction branches','log implied variance\nanalytic price transform'],
          ['Same existing checkpoint\nfrozen in new-branch arms','Train-only affine conditioning\nsame 26 branch inputs',
           'Published modified MLP\nU,V encoders; 5 gated layers' if 'MODIFIED' in selected else 'Selected: '+SHORT.get(selected,selected),
           'Add bounded correction\nsame price + PDE transform']]
    if selected=='A0_CONTINUE':
        rows[1]=['Existing C3 core\nweights frozen','Existing v5 features\nno new input transform','Resume 3 shared branches\nsame 5,859 trainable weights','Same output transform\nno added parameters']
    for ax,blocks,title in zip(axes,rows,['Reproduced baseline','Selected architecture: '+selected]):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title)
        for i,txt in enumerate(blocks):
            x=.015+i*.25;rect=FancyBboxPatch((x,.28),.215,.4,boxstyle='round,pad=.01',facecolor='#e9f0f7',edgecolor='#326a96');ax.add_patch(rect)
            ax.text(x+.1075,.48,txt,ha='center',va='center',fontsize=9)
            if i<3:ax.annotate('',xy=(x+.245,.48),xytext=(x+.22,.48),arrowprops={'arrowstyle':'->'})
    if 'MODIFIED' in selected:axes[1].text(.5,.07,'U=tanh(Wu f+bu), V=tanh(Wv f+bv); H←tanh(W H+b)⊙U + (1−tanh(W H+b))⊙V',ha='center',fontsize=9)
    fig.savefig(dest/'architecture_diagrams.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    training_plots();metrics_plots()
    selected=read(OUT/'selection.json')['chosen'];diagram(selected or 'NONE')
