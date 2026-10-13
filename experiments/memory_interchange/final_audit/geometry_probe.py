"""Descriptions of native coordinates and fixed T-only context probes."""
import itertools
import json
import numpy as np
from ..geometry import linear_cka, raw_distance, ridge_fit, ridge_apply
from .common import *

def describe(v):
    v=np.asarray(v,float)
    return {'n':len(v),'min':float(v.min()),'q05':float(np.quantile(v,.05)),
            'q25':float(np.quantile(v,.25)),'median':float(np.median(v)),
            'q75':float(np.quantile(v,.75)),'q95':float(np.quantile(v,.95)),'max':float(v.max())}

def geometry():
    rows=[];old_lookup={}
    for seed in SEEDS:
        root=RUNS/f'seed_{seed}/analysis';cache=dict(np.load(root/'geometry_states.npz'))
        original={(int(r['a']),int(r['b']),int(r['d'])):r for r in csv_read(root/'geometry.csv')}
        for di,d in enumerate((0,10)):
            for a,b in itertools.combinations(range(18),2):
                x=cache['memory'][di,a].astype(np.float64);y=cache['memory'][di,b].astype(np.float64)
                mx,my=x.mean(0),y.mean(0)
                den=.5*np.mean(np.sum(x*x+y*y,axis=-1))+1e-12
                mean=float(np.sum((mx-my)**2)/den)
                center=float(np.mean(np.sum(((x-mx)-(y-my))**2,axis=-1))/den)
                raw=raw_distance(x,y);cka=linear_cka(x,y);old=original[(a,b,d)]
                np.testing.assert_allclose(raw,mean+center,atol=2e-14,rtol=2e-13)
                np.testing.assert_allclose(raw,float(old['raw']),atol=2e-14,rtol=2e-13)
                np.testing.assert_allclose(cka,float(old['cka']),atol=2e-13,rtol=2e-13)
                row={'seed':seed,'a':a,'b':b,'d':d,'cross':int(CONDITIONS[a][0]!=CONDITIONS[b][0]),
                     'cka':cka,'D_raw':raw,'D_mean':mean,'D_center':center,'denominator':den,
                     'decomposition_error':raw-mean-center,'mean_fraction':mean/raw if raw else 0.,
                     'raw_mean_difference_L2':float(np.linalg.norm(mx-my)),
                     'control_distance':float(old['control_distance'])}
                rows.append(row);old_lookup[(seed,a,b,d)]=row
    csv_write(OUT/'geometry_decomposition.csv',rows)
    distributions=[]
    for seed in ['all',*SEEDS]:
        for cross in ['all',0,1]:
            for d in ['all',0,10]:
                selected=[r for r in rows if (seed=='all' or r['seed']==seed) and
                          (cross=='all' or r['cross']==cross) and (d=='all' or r['d']==d)]
                for key in ('cka','D_raw','D_mean','D_center','mean_fraction','control_distance'):
                    distributions.append({'seed':seed,'cross':cross,'d':d,'metric':key,**describe([r[key] for r in selected])})
    csv_write(OUT/'geometry_distributions.csv',distributions)
    examples=[]
    for r in csv_read(OLD/'prediction/counterexamples.csv'):
        seed,a,b,d=map(lambda k:int(float(r[k])),('seed','a','b','d'))
        nativ=[nr for nr in csv_read(RUNS/f'seed_{seed}/analysis/native_strata.csv') if int(nr['d'])==d and nr['S']==r['S'] and int(nr['condition']) in (a,b)]
        g=old_lookup[(seed,a,b,d)]
        examples.append({**r,'actual_D_mean':g['D_mean'],'actual_D_center':g['D_center'],
            'native_min_direction_query_accuracy':min(float(nr['accuracy']) for nr in nativ),
            'r_a':CONDITIONS[a][0],'H_a':CONDITIONS[a][1],'r_b':CONDITIONS[b][0],'H_b':CONDITIONS[b][1]})
    csv_write(OUT/'original_fixed_counterexamples.csv',examples)
    # An independent mathematical check, not fitted to the observed states.
    rng=np.random.default_rng(20261019);x=rng.normal(size=(40,9));v=rng.normal(size=9)
    shift=linear_cka(x,x+v);np.testing.assert_allclose(shift,1.,atol=1e-14)
    write_json(OUT/'geometry_checks.json',{'unique_geometry_rows':len(rows),'deduplicated_across_S':True,
        'max_decomposition_error':max(abs(r['decomposition_error']) for r in rows),
        'old_fixed_counterexamples_preserved':len(examples),'synthetic_translation_CKA':shift,
        'synthetic_translation_is_not_claim_about_RNN':True})

def states_rows(cache,component,delays,nonmixed):
    x=[];y=[];meta=[]
    for di,d in enumerate((0,10)):
        if d not in delays:continue
        for a,(r,h) in enumerate(CONDITIONS):
            if nonmixed is not None and (h not in ('AB','BA')) != nonmixed:continue
            xx=cache['memory'][di,a] if component=='m' else cache['control'][di,a:a+1]
            x.extend(xx);y.extend([2*r-1]*len(xx))
            meta.extend([{'r':r,'H':h,'d':d,'condition':a,'q_index':i if component=='m' else -1} for i in range(len(xx))])
    return np.asarray(x,np.float64),np.asarray(y,np.float64),meta

def probes():
    summaries=[];contrasts=[];inventory=[]
    for seed in SEEDS:
        root=RUNS/f'seed_{seed}/analysis';out=OUT/'context_probes'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
        caches={'T':dict(np.load(root/'probe_T_states.npz')),'G':dict(np.load(root/'geometry_states.npz'))}
        for component in ('m','c'):
            for variant,delays in [('pooled',[0,10]),('fit_d0',[0]),('fit_d10',[10])]:
                x,y,meta=states_rows(caches['T'],component,delays,True)
                mean=x.mean(0);scale=x.std(0);scale[scale==0]=1.
                z=(x-mean)/scale;fit=ridge_fit(z,y[:,None],1.)
                np.savez_compressed(out/f'{component}_{variant}_fit.npz',mean=mean,scale=scale,**fit,
                    labels=y,fit_conditions=np.array([[m['r'],CONDITIONS.index((m['r'],m['H'])),m['d']] for m in meta]),
                    fit_content=caches['T']['content'],alpha=1.,threshold=0.)
                inventory.append({'seed':seed,'component':component,'variant':variant,'fitting_rows':len(x),
                    'distinct_r_H_d':len(set((m['r'],m['H'],m['d']) for m in meta)),
                    'actual_distinct_state_vectors':len(np.unique(x,axis=0)),
                    'nonzero_coordinate_std_min':float(np.min(x.std(0)[x.std(0)>0])) if np.any(x.std(0)>0) else 0.,
                    'coordinate_std_median':float(np.median(x.std(0))),'coordinate_std_max':float(x.std(0).max()),
                    'T_only':True,'AB_BA_excluded':True})
                for split,cache in caches.items():
                    xx,yy,mm=states_rows(cache,component,[0,10],None)
                    score=ridge_apply((xx-mean)/scale,fit)[:,0]
                    predicted=np.where(score>=0,1,-1);margin=yy*score
                    # Save every margin; c has one row per independent state.
                    np.savez_compressed(out/f'{component}_{variant}_{split}_outputs.npz',score=score,margin=margin,
                        labels=yy,predicted=predicted,condition=np.array([m['condition'] for m in mm]),
                        d=np.array([m['d'] for m in mm]),q_index=np.array([m['q_index'] for m in mm]),
                        content=cache['content'],orbit_id=cache['orbit_id'])
                    for d in (0,10):
                        for hgroup in ('nonmixed','AB_BA',*[h for r,h in CONDITIONS[:9]]):
                            mask=np.array([m['d']==d and (m['H'] not in ('AB','BA') if hgroup=='nonmixed' else
                                          m['H'] in ('AB','BA') if hgroup=='AB_BA' else m['H']==hgroup) for m in mm])
                            summaries.append({'seed':seed,'component':component,'variant':variant,'split':split,'d':d,
                              'delay_relation':'pooled' if variant=='pooled' else ('same' if d in delays else 'different'),
                              'history_group':hgroup,'n':int(mask.sum()),'accuracy':float((predicted[mask]==yy[mask]).mean()),
                              'signed_margin_mean':float(margin[mask].mean()),'signed_margin_min':float(margin[mask].min()),
                              'signed_margin_q05':float(np.quantile(margin[mask],.05)),
                              'signed_margin_median':float(np.median(margin[mask]))})
            for split,cache in caches.items():
                for di,d in enumerate((0,10)):
                    for a,(r,h) in enumerate(CONDITIONS[:9]):
                        x=cache['memory'][di,a].astype(float) if component=='m' else cache['control'][di,a:a+1].astype(float)
                        y=cache['memory'][di,a+9].astype(float) if component=='m' else cache['control'][di,a+9:a+10].astype(float)
                        l2=np.linalg.norm(x-y,axis=-1)
                        contrasts.append({'seed':seed,'component':component,'split':split,'d':d,'H':h,'n':len(l2),
                             'paired_context_L2_mean':float(l2.mean()),'paired_context_L2_min':float(l2.min()),
                             'paired_context_L2_max':float(l2.max()),
                             'context_mean_L2':float(np.linalg.norm(x.mean(0)-y.mean(0)))})
        print(json.dumps({'phase':'context probes','seed':seed}),flush=True)
    csv_write(OUT/'context_probe_results.csv',summaries);csv_write(OUT/'context_raw_contrasts.csv',contrasts)
    csv_write(OUT/'context_probe_fit_inventory.csv',inventory)
    write_json(OUT/'context_probe_summary.json',{'alpha':1,'threshold':0,'training_split':'T only, 512 fixed original contents',
        'history_holdout':['AB','BA'],'c_no_content_replication':True,'no_probe_selection':True,
        'standardization':'T-only; tiny nonzero signals are scaled too; decoding is not a causal-use test',
        'fit_count':len(inventory),'result_rows':len(summaries),'no_patches_or_B_labels_used':True})

def run():
    geometry();probes()

if __name__=='__main__':run()
