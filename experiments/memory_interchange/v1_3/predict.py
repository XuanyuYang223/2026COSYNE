import csv
import json
import time
import numpy as np
from ..common import write_json
from ..predict import (base_features,fit_predict,segment,errors,SEGMENTS,save_csv,context_category)
from scipy.stats import spearmanr
from .common import RESULTS,RUNS,config

MODELS={'M0_old':(),'M0_old_CKA':('cka',),'M0_v13':('D_input',),
        'M_CKA_v13':('D_input','cka'),'M_raw_v13':('D_input','raw'),
        'M_proc_v13':('D_input','proc'),'M_init_v13':('D_input','init_cka'),
        'M_init_CKA_v13':('D_input','init_cka','cka')}
MODES=('all','rec','both')

def merged_rows(runs):
    rows=[]
    for run in runs:
        with (run/'analysis/geometry.csv').open() as f:geo=list(csv.DictReader(f))
        index={(int(r['a']),int(r['b']),int(r['d'])):r for r in geo}
        with (run/'analysis/labels.csv').open() as f:
            for r in csv.DictReader(f):
                g=index[(int(r['a']),int(r['b']),int(r['d']))]
                rows.append({k:(v if k=='S' else float(v) if v else float('nan')) for k,v in {**g,**r}.items()})
    return rows

def design(rows,model):
    return np.array([base_features(r)+[r[k] for k in MODELS[model]] for r in rows],np.float64)

def folds(rows,seeds,mode):
    predictions=[];scores=[];coefficients={}
    for held in seeds:
        train=[r for r in rows if int(r['seed'])!=held and not r['unseen_pair'] and np.isfinite(r[f'matched_L_{mode}'])]
        test=[r for r in rows if int(r['seed'])==held]
        yt=np.array([r[f'matched_L_{mode}'] for r in train]);y=np.array([r[f'matched_L_{mode}'] for r in test])
        pred={}
        for model in MODELS:
            pred[model],prep,fit=fit_predict(design(train,model),yt,design(test,model),1.)
            coefficients[f'{mode}_{held}_{model}']={**{k:v.tolist() for k,v in prep.items()},**{k:v.tolist() for k,v in fit.items()},
                    'fitting_seeds':[s for s in seeds if s!=held],'fitting_pairs':91,'finite_fitting_rows':len(train)}
        for i,r in enumerate(test):
            predictions.append({'seed':held,'a':int(r['a']),'b':int(r['b']),'d':int(r['d']),'S':r['S'],
                                'mode':mode,'segment':segment(r),'quality':int(r['quality']),'observed':y[i],
                                **{model:pred[model][i] for model in MODELS}})
    for held in [*seeds,'overall']:
        test=[r for r in predictions if held=='overall' or r['seed']==held]
        for seg in SEGMENTS:
            for qbin in ('all',0,1,2):
                selected=[r for r in test if r['segment']==seg and (qbin=='all' or r['quality']==qbin) and np.isfinite(r['observed'])]
                if not selected:continue
                y=np.array([r['observed'] for r in selected]);p0=np.array([r['M0_v13'] for r in selected]);old=np.array([r['M0_old'] for r in selected])
                b=errors(y,p0);bo=errors(y,old)
                for model in MODELS:
                    e=errors(y,np.array([r[model] for r in selected]))
                    scores.append({'seed':held,'mode':mode,'segment':seg,'quality':qbin,'model':model,'n':len(selected),**e,
                                   'mae_improvement':b['mae']-e['mae'],'rmse_improvement':b['rmse']-e['rmse'],
                                   'mae_improvement_vs_old':bo['mae']-e['mae'],'rmse_improvement_vs_old':bo['rmse']-e['rmse']})
    return predictions,scores,coefficients

def bootstrap_labels(rows,blocks,weights,mi):
    numerator=(blocks['loss_sums'][...,mi]*weights[None,None,None,:]).sum(-1)
    denominator=(blocks['mask_counts'][...,mi]*weights[None,None,None,:]).sum(-1)
    lookup={(int(d),str(s)):j for j,(d,s) in enumerate(zip(blocks['delays'],blocks['suffixes']))}
    y=[]
    for r in rows:
        j=lookup[(int(r['d']),r['S'])];a,b=int(r['a']),int(r['b'])
        num=numerator[j,a,b]+numerator[j,b,a];den=denominator[j,a,b]+denominator[j,b,a]
        y.append(num/den if den else np.nan)
    return np.array(y)

def bootstrap(rows,seeds):
    cfg=config();by={s:[r for r in rows if int(r['seed'])==s] for s in seeds}
    blocks={s:dict(np.load(RUNS/f'seed_{s}/analysis/orbit_blocks.npz')) for s in seeds}
    x={s:{m:design(by[s],m) for m in MODELS} for s in seeds}
    seen={s:np.array([not r['unseen_pair'] for r in by[s]]) for s in seeds}
    segments={s:np.array([segment(r) for r in by[s]]) for s in seeds}
    rng=np.random.default_rng(20261016);samples=[]
    for rep in range(cfg['prediction_v13']['bootstrap_replicates']):
        selected=rng.choice(seeds,len(seeds),replace=True)
        if len(set(selected))<2:continue
        weights={s:np.bincount(rng.integers(len(blocks[s]['orbit_ids']),size=len(blocks[s]['orbit_ids'])),minlength=len(blocks[s]['orbit_ids'])) for s in set(selected)}
        for mi,mode in enumerate(MODES):
            y={s:bootstrap_labels(by[s],blocks[s],weights[s],mi) for s in set(selected)}
            pred={}
            for held in set(selected):
                fitseeds=[int(s) for s in selected if s!=held]
                fitmask={s:seen[s]&np.isfinite(y[s]) for s in fitseeds}
                yt=np.concatenate([y[s][fitmask[s]] for s in fitseeds]);pred[held]={}
                for m in MODELS:
                    xt=np.concatenate([x[s][m][fitmask[s]] for s in fitseeds])
                    pred[held][m]=fit_predict(xt,yt,x[held][m],1.)[0]
            for seg in SEGMENTS:
                yy=np.concatenate([y[int(s)][segments[int(s)]==seg] for s in selected])
                pp={m:np.concatenate([pred[int(s)][m][segments[int(s)]==seg] for s in selected]) for m in MODELS}
                valid=np.isfinite(yy);yy=yy[valid];pp={m:p[valid] for m,p in pp.items()}
                baseline=errors(yy,pp['M0_v13']);old=errors(yy,pp['M0_old']);init=errors(yy,pp['M_init_v13'])
                for m in MODELS:
                    e=errors(yy,pp[m])
                    samples.append({'replicate':rep,'mode':mode,'segment':seg,'model':m,
                                    'mae_improvement':baseline['mae']-e['mae'],'rmse_improvement':baseline['rmse']-e['rmse'],
                                    'mae_improvement_vs_old':old['mae']-e['mae'],'rmse_improvement_vs_old':old['rmse']-e['rmse'],
                                    'mae_improvement_vs_init':init['mae']-e['mae'],'rmse_improvement_vs_init':init['rmse']-e['rmse']})
        if rep%100==0:print(json.dumps({'bootstrap_replicate':rep}),flush=True)
    intervals=[]
    metrics=('mae_improvement','rmse_improvement','mae_improvement_vs_old','rmse_improvement_vs_old','mae_improvement_vs_init','rmse_improvement_vs_init')
    for mode in MODES:
        for seg in SEGMENTS:
            for model in MODELS:
                v=[r for r in samples if r['mode']==mode and r['segment']==seg and r['model']==model]
                intervals.append({'mode':mode,'segment':seg,'model':model,'replicates':len(v),
                                  **{k+'_95ci':np.quantile([r[k] for r in v],[.025,.975]).tolist() for k in metrics}})
    return samples,intervals

def run():
    out=RESULTS/'prediction';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    start=time.perf_counter();seeds=config()['formal_seeds']
    rows=merged_rows([RUNS/f'seed_{s}' for s in seeds])
    assert sorted(set(int(r['seed']) for r in rows))==seeds and len(rows)==6*918
    save_csv(out/'full_distribution.csv',rows)
    p=[];scores=[];coef={}
    for mode in MODES:
        pred,ss,cc=folds(rows,seeds,mode);p+=pred;scores+=ss;coef.update(cc)
    save_csv(out/'predictions.csv',p);save_csv(out/'scores.csv',scores);write_json(out/'coefficients.json',coef)
    samples,ci=bootstrap(rows,seeds);save_csv(out/'bootstrap.csv',samples);write_json(out/'bootstrap_intervals.json',ci)
    correlations=[];examples=[]
    for seed in seeds:
        for d in (0,10):
            for suffix in ('NN','NA','NB'):
                for category in ((0,0),(0,1),(1,1)):
                    group=[r for r in rows if int(r['seed'])==seed and r['d']==d and r['S']==suffix and context_category(r)==category]
                    finite=[r for r in group if np.isfinite(r['cka'])]
                    x=np.array([r['cka'] for r in finite]);y=np.array([r['loss_positive'] for r in finite])
                    defined=len(x)>=3 and np.var(x)>1e-12 and np.var(y)>1e-12
                    correlations.append({'seed':seed,'d':d,'S':suffix,'context':category,'n':len(x),
                                         'spearman':float(spearmanr(x,y).statistic) if defined else None})
                    if finite:
                        xc=np.quantile(x,.75);rc=np.quantile([r['new_error'] for r in finite],.75)
                        examples.extend({**r,'cka_cutoff':float(xc),'R_cutoff':float(rc)} for r in finite if r['cka']>=xc and r['new_error']>=rc and r['new_error']>=.005)
    write_json(out/'stratified_spearman.json',correlations);save_csv(out/'counterexamples.csv',examples)
    write_json(out/'summary.json',{'rows':len(rows),'prediction_rows':len(p),'scores':scores,'intervals':ci,
               'counterexample_count':len(examples),'CPU_wall_seconds':time.perf_counter()-start,
               'all_six_seeds_preserved':seeds,'primary_model':'M_CKA_v13','primary_label':'L_all'})

if __name__=='__main__':run()
