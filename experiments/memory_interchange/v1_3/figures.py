import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import RESULTS,RUNS

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not fig.get_constrained_layout():fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)

def intervention_figures(out,distribution):
    r=distribution;folder=out/'figures'
    for feature in ('cka','raw','proc'):
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for di,d in enumerate((0,10)):
            for si,s in enumerate(('NN','NA','NB')):
                ax=axes[di,si]
                for cross,marker,label in ((0,'o','same'),(1,'^','cross')):
                    g=[v for v in r if int(float(v['d']))==d and v['S']==s and int(float(v['cross_context']))==cross]
                    art=ax.scatter([float(v[feature]) for v in g],[float(v['loss_positive']) for v in g],
                        c=[float(v['native_correct']) for v in g],cmap='viridis',vmin=0,vmax=1,marker=marker,s=12,alpha=.5,label=label)
                ax.set(xlabel=feature,ylabel='L_all (nat)',title=f'd={d}, S={s}');ax.legend(fontsize=8)
        fig.colorbar(art,ax=axes,label='Native pair accuracy');save(fig,folder/f'{feature}_all_damage')
    # Keep rates, CE, and coverage on separate axes to avoid mixing units.
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for mi,mode in enumerate(('all','rec','both')):
        group=r;axes[0].plot([0,1],[np.mean([float(v[f'matched_R_{mode}']) for v in group]),np.mean([float(v[f'wrong_R_{mode}']) for v in group])],'o-',label=mode)
        axes[1].plot([0,1],[np.mean([float(v[f'matched_L_{mode}']) for v in group]),np.mean([float(v[f'wrong_L_{mode}']) for v in group])],'o-',label=mode)
        axes[2].bar(mi,np.mean([float(v[f'coverage_{mode}']) for v in group]))
    axes[0].set(xticks=(0,1),xticklabels=('matched','wrong'),ylabel='New error rate; mode-specific denominator')
    axes[1].set(xticks=(0,1),xticklabels=('matched','wrong'),ylabel='Positive CE harm (nat)')
    axes[2].set(xticks=(0,1,2),xticklabels=('all','rec','both'),ylabel='Retained fraction')
    for ax in axes[:2]:ax.legend()
    save(fig,folder/'behavior_and_coverage')

def training():
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed in range(101,107):
        p=RUNS/f'seed_{seed}/stage_7/curve.json'
        if not p.exists():continue
        r=json.loads(p.read_text())
        axes[0].plot([v['step'] for v in r],[100*v['V_accuracy'] for v in r],label=str(seed))
        axes[1].plot([v['step'] for v in r],[100*v['V_strata']['non_N']['3']['accuracy'] for v in r],label=str(seed))
    axes[0].axhline(97,c='black',ls='--');axes[1].axhline(95,c='black',ls='--')
    for ax in axes:ax.set(xlabel='Stage 7 updates',ylabel='Accuracy (%)');ax.legend(fontsize=8)
    axes[0].set_title('Original 8,192 V');axes[1].set_title('V k=3; fine operations reported separately')
    save(fig,RESULTS/'figures/formal_learning')

def prediction():
    r=rows(RESULTS/'prediction/scores.csv');models=('M0_old','M0_old_CKA','M0_v13','M_CKA_v13','M_raw_v13','M_proc_v13','M_init_v13','M_init_CKA_v13')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axes,('mae','rmse')):
        for seed in range(101,107):
            vals=[next(v for v in r if v['seed']==str(seed) and v['mode']=='all' and v['segment']=='unseen_cross' and v['quality']=='all' and v['model']==m) for m in models]
            ax.plot(range(len(models)),[float(v[metric]) for v in vals],'.-',label=str(seed))
        ax.set(xticks=range(len(models)),xticklabels=models,ylabel=metric.upper()+' (nat)',title='Held seed + unseen cross-context pair')
        ax.tick_params(axis='x',rotation=60);ax.legend(fontsize=8)
    save(fig,RESULTS/'figures/held_out_prediction')
