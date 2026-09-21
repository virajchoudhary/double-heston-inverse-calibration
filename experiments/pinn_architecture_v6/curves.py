"""Common fixed-grid price curves, true spot derivatives and surface plots."""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from .evaluate import nets
from .train import ARMS
from experiments.nifty_multifactor_v4.literature_exact import exact,black


def compute():
    dest=OUT/'curves';dest.mkdir(exist_ok=True)
    cfg=read(old.HERE/'config.json');p=np.array(cfg['published_double_slow_first'])
    x=np.linspace(-.36,.36,81);tau=np.geomspace(7/365,2.,41)
    xx,tt=np.meshgrid(x,tau);xf,tf=xx.ravel(),tt.ravel()
    # r=q=0, K=1, S=exp(x): convert forward-normalized exact price to C/K.
    teacher=exact(p,xf,tf)*np.exp(xf)
    np.savez_compressed(dest/'teacher.npz',x=x,tau=tau,price=teacher.reshape(xx.shape),params=p)
    z=np.column_stack([xf,np.full(len(xf),p[4]),np.full(len(xf),p[9]),tf]);sp=np.tile(p,(len(z),1))
    rows=[];available=['BASELINE']+[arm for arm in ARMS if all((OUT/arm/f'seed_{s}/completed.json').exists() for s in SEEDS)]
    for arm in available:
        models=nets(arm);values=[];delta=[];gamma=[]
        for start in range(0,len(z),128):
            coords=T(z[start:start+128]).requires_grad_(True);st=old.structural(sp[start:start+128])
            price=torch.stack([net.price(coords,st) for net in models]).mean(0)
            dx=torch.autograd.grad(price.sum(),coords,create_graph=True)[0][:,0]
            dxx=torch.autograd.grad(dx.sum(),coords)[0][:,0]
            values.extend(price.detach().numpy());delta.extend((dx*torch.exp(-coords[:,0])).detach().numpy())
            gamma.extend(((dxx-dx)*torch.exp(-2*coords[:,0])).detach().numpy())
        pred=np.array(values);delta=np.array(delta);gamma=np.array(gamma);error=pred-teacher
        z0=T(z[:81].copy());z0[:,-1]=0;st=old.structural(sp[:81])
        with torch.no_grad():terminal=torch.stack([net.price(z0,st) for net in models]).mean(0)
        terminal_error=float((terminal-torch.clamp_min(torch.expm1(z0[:,0]),0)).abs().max())
        edge=np.tile((np.arange(81)==0)|(np.arange(81)==80),41)
        rows.append({'arm':arm,'price_RMSE_C_over_K':float(np.sqrt(np.mean(error**2))),
            'max_error_C_over_K':float(abs(error).max()),'monotonicity_violation_percent':100*float((delta<-1e-8).mean()),
            'convexity_violation_percent':100*float((gamma<-1e-8).mean()),'terminal_max_error':terminal_error,
            'finite_edge_teacher_RMSE':float(np.sqrt(np.mean(error[edge]**2))),
            'call_bound_violations':int(((pred<np.maximum(np.expm1(xf),0)-1e-9)|(pred>np.exp(xf)+1e-9)).sum())})
        np.savez_compressed(dest/f'{arm}.npz',price=pred.reshape(xx.shape),error=error.reshape(xx.shape),
            relative_error=(abs(error)/np.maximum(abs(teacher),1e-8)).reshape(xx.shape),delta=delta,gamma=gamma)
    pd.DataFrame(rows).to_csv(dest/'curve_quality.csv',index=False)
    plot(available,x,tau,teacher.reshape(xx.shape),dest,p)


def plot(arms,x,tau,teacher,dest,p):
    xx,tt=np.meshgrid(np.exp(x),tau*365)
    max_error=max(float(np.max(abs(np.load(dest/f'{arm}.npz')['error']))) for arm in arms)
    max_relative=max(float(np.max(np.load(dest/f'{arm}.npz')['relative_error'])) for arm in arms)
    for arm in arms:
        a=np.load(dest/f'{arm}.npz');pred=a['price']
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for index in [0,12,26,40]:
            line=axes[0].plot(np.exp(x),teacher[index],label=f'exact {tau[index]*365:.0f}d')[0]
            axes[0].plot(np.exp(x),pred[index],ls='--',color=line.get_color())
        for index in [20,40,60]:
            line=axes[1].plot(tau*365,teacher[:,index],label=f'exact S/K={np.exp(x[index]):.2f}')[0]
            axes[1].plot(tau*365,pred[:,index],ls='--',color=line.get_color())
        # Qualitative BS curve at instantaneous total-volatility anchor only.
        axes[0].plot(np.exp(x),black(x,np.full(len(x),tau[12]),np.sqrt(p[4]+p[9]))*np.exp(x),c='gray',ls=':',label='BS shape reference')
        axes[0].set(xlabel='S/K (r=q=0)',ylabel='C/K',title='Solid: exact DH; dashed: PINN');axes[0].legend(fontsize=7)
        axes[1].set(xlabel='Maturity (days)',ylabel='C/K',title='Same fixed state and teacher');axes[1].legend(fontsize=7)
        fig.suptitle(arm);fig.savefig(dest/f'{arm}_curves.png',dpi=140);plt.close(fig)
        fig=plt.figure(figsize=(12,8),layout='constrained')
        for j,(title,values,upper) in enumerate([('Exact Double Heston',teacher,teacher.max()),('PINN',pred,teacher.max()),
                ('Absolute error',abs(a['error']),max_error),('Relative error; denominator floor 1e-8',a['relative_error'],max_relative)]):
            ax=fig.add_subplot(2,2,j+1,projection='3d');ax.plot_surface(xx,tt,values,cmap='viridis',vmin=0,vmax=upper)
            ax.set(xlabel='S/K',ylabel='Days',title=title,zlim=(0,upper*1.01));ax.view_init(25,-130)
        fig.suptitle(arm+' · identical grid and scales across candidates');fig.savefig(dest/f'{arm}_surfaces.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        im=ax.pcolormesh(np.exp(x),tau*365,abs(a['error']),shading='auto',vmin=0,vmax=max_error,cmap='magma')
        ax.set(yscale='log',xlabel='S/K',ylabel='Days',title=arm+' absolute C/K error');fig.colorbar(im,ax=ax)
        fig.savefig(dest/f'{arm}_heatmap.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    torch.set_num_threads(1);compute()
