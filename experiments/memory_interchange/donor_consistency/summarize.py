"""Counts, distributions and fixed-six-model orbit-block uncertainty."""
import itertools
import time
import numpy as np
from .common import *
from .analyze import bool_flag

def selectors(plan,context,history,S,d,treatment,kind):
    conditions=plan['conditions']
    pairs=[i for i,(a,b) in enumerate(PAIRS) if
        (context=='all' or (conditions[a]['r']!=conditions[b]['r'])==(context=='cross')) and
        (history=='all' or (conditions[a]['H']==conditions[b]['H'])==(history=='same'))]
    cells=[i for i in range(6) if (S=='all' or SUFFIXES[i%3]==S) and (d=='all' or DELAYS[i//3]==int(d))]
    treatments=[0,1] if treatment=='all' else [TREATMENTS.index(treatment)]
    kinds=[0,1] if kind=='all' else [QUERY_TYPES.index(kind)]
    return cells,treatments,pairs,kinds

def tensors(dataset,seed):
    root=OUT/'data'/dataset/f'seed_{seed}'
    return {name:np.load(root/(file+'.npy'),mmap_mode='r') for name,file in
            [('flags','flags'),('diff','probability_differences'),('answers','answers')]}

def selected(bundle,selector):
    cells,ts,pairs,kinds=selector
    index=np.ix_(cells,ts,pairs,np.arange(bundle['flags'].shape[3]),kinds,np.arange(7))
    flags=bundle['flags'][index]
    dif=bundle['diff'][index]
    ans=bundle['answers'][index]
    return flags,dif,ans

def rate(num,den):return float(num/den) if den else ''

def measure(flags,diff,answers,donor_subset):
    av=bool_flag(flags,'available');eligible=av.copy()
    if donor_subset=='correct':eligible&=bool_flag(flags,'donor_correct')
    elif donor_subset=='error':eligible&=~bool_flag(flags,'donor_correct')
    agreement=bool_flag(flags,'agree')&eligible
    pv=bool_flag(flags,'probability_available')&eligible
    n=int(eligible.sum());nprob=int(pv.sum())
    counts=eligible.sum(axis=(3,5));numerator=agreement.sum(axis=(3,5))
    values=np.divide(numerator,counts,out=np.full(counts.shape,np.nan,float),where=counts>0)
    row={'n_expected_queries':int(flags.size),'n_available_queries':int(av.sum()),'n_missing_queries':int((~av).sum()),
        'n_subgroup_queries':n,'answer_agree_n':int(agreement.sum()),'answer_disagree_n':n-int(agreement.sum()),
        'answer_agreement_pooled':rate(agreement.sum(),n),
        'answer_agreement_equal_cells':float(np.nanmean(values)) if np.isfinite(values).any() else '',
        'nonempty_direction_dS_type_cells':int((counts>0).sum()),'probability_n':nprob,
        'probability_missing_in_subgroup':n-nprob}
    for name in ('patch_tie','donor_tie'):
        known=bool_flag(flags,name+'_known')&eligible
        row[name+'_known_n']=int(known.sum());row[name+'_n']=int((bool_flag(flags,name)&eligible).sum())
    for flag,label in [('exact','probability_exact_query'),('close_1e6','probability_close_1e6'),
                       ('close_1e5','probability_close_1e5'),('close_1e4','probability_close_1e4')]:
        num=int((bool_flag(flags,flag)&pv).sum())
        row[label+'_n']=num;row[label+'_rate']=rate(num,nprob)
    element_n=int(answers[...,3][pv].astype(np.int64).sum()) if nprob else 0
    row['probability_exact_element_n']=element_n;row['probability_element_n']=7*nprob
    row['probability_exact_element_rate']=rate(element_n,7*nprob)
    for mi,label in enumerate(('max_abs_probability_difference','TV')):
        v=diff[...,mi][pv]
        if nprob:
            qq=np.quantile(v,[.5,.95,.99])
            weighted_sum=np.where(pv,diff[...,mi],0.).sum(axis=(3,5))
            pcounts=pv.sum(axis=(3,5));means=np.divide(weighted_sum,pcounts,out=np.full(pcounts.shape,np.nan,float),where=pcounts>0)
            row.update({label+'_mean_pooled':float(v.mean()),label+'_mean_equal_cells':float(np.nanmean(means)),
                label+'_median':float(qq[0]),label+'_q95':float(qq[1]),label+'_q99':float(qq[2]),label+'_max':float(v.max())})
        else:
            for suffix in ('mean_pooled','mean_equal_cells','median','q95','q99','max'):row[label+'_'+suffix]=''
    # Vector consistency is measured on ALL original query predictions; the
    # donor correctness split is query-level and cannot define a full vector.
    if donor_subset=='all':
        all14=av.all(axis=(-1,-2));eq14=bool_flag(flags,'agree').all(axis=(-1,-2))&all14
        row['full14_available_n']=int(all14.sum());row['full14_agree_n']=int(eq14.sum());row['full14_agreement']=rate(eq14.sum(),all14.sum())
        for kind,name in enumerate(('item7','position7')):
            if flags.shape[-2]==2:
                v=av[...,kind,:].all(-1);eq=bool_flag(flags,'agree')[...,kind,:].all(-1)&v
                row[name+'_available_n']=int(v.sum());row[name+'_agree_n']=int(eq.sum());row[name+'_agreement']=rate(eq.sum(),v.sum())
            else:row[name+'_available_n']='';row[name+'_agree_n']='';row[name+'_agreement']=''
        # Only a selection containing BOTH query types has a 14-vector.
        if flags.shape[-2]!=2:row['full14_available_n']='';row['full14_agree_n']='';row['full14_agreement']=''
        if flags.shape[-2]==2:
            for ai,name in [(0,'patch'),(1,'donor')]:
                forward=answers[...,0,:,ai];reverse=answers[...,1,:,ai]
                fbij=np.all(np.sort(forward,axis=-1)==np.arange(7),axis=-1)
                rbij=np.all(np.sort(reverse,axis=-1)==np.arange(7),axis=-1)
                safe=np.minimum(forward,6)
                mutual=fbij&rbij&np.all(np.take_along_axis(reverse,safe,axis=-1)==np.arange(7),axis=-1)
                row[name+'_nonbijective_or_noninverse_vector_n']=int((all14&~mutual).sum())
    else:
        for key in ('full14_available_n','full14_agree_n','full14_agreement','item7_available_n','item7_agree_n','item7_agreement',
                    'position7_available_n','position7_agree_n','position7_agreement','patch_nonbijective_or_noninverse_vector_n','donor_nonbijective_or_noninverse_vector_n'):
            row[key]=''
    row['row_status']='EMPTY' if flags.size==0 else 'NO_SUBGROUP_OBSERVATIONS' if n==0 else 'ANSWER_ONLY' if nprob==0 else 'OK'
    return row

def merge_seed_rows(rows,bundles,selector,subgroup):
    result={}
    count_columns=[k for k in rows[0] if k.endswith('_n') or k.startswith('n_') or k in ('nonempty_direction_dS_type_cells','answer_agree_n','answer_disagree_n','probability_n','probability_missing_in_subgroup')]
    for k in count_columns:
        vals=[r[k] for r in rows if r.get(k,'')!='']
        result[k]=sum(vals) if vals else ''
    # Equal-model averages of probabilities/rates; pooled descriptive quantiles
    # are recomputed from all actual observations (never averaged medians).
    for k in rows[0]:
        if k in count_columns or k=='row_status':continue
        vals=[r[k] for r in rows if r.get(k,'')!='']
        if not vals:result[k]=''
        elif k.endswith('_max'):result[k]=max(vals)
        elif k.endswith(('_median','_q95','_q99')):continue
        elif k=='answer_agreement_pooled':result[k]=rate(result['answer_agree_n'],result['n_subgroup_queries'])
        elif k.endswith('_rate') and k.startswith('probability_'):
            den=result['probability_element_n'] if k=='probability_exact_element_rate' else result['probability_n']
            key=k[:-5]+'_n';result[k]=rate(result[key],den)
            result[k+'_equal_seed']=float(np.mean(vals))
        else:result[k]=float(np.mean(vals))
    pooled=[[],[]]
    for b in bundles:
        f,d,a=selected(b,selector);keep=bool_flag(f,'available')&bool_flag(f,'probability_available')
        if subgroup=='correct':keep&=bool_flag(f,'donor_correct')
        elif subgroup=='error':keep&=~bool_flag(f,'donor_correct')
        for mi in (0,1):pooled[mi].append(d[...,mi][keep])
    for mi,label in enumerate(('max_abs_probability_difference','TV')):
        v=np.concatenate(pooled[mi]) if pooled[mi] else np.empty(0)
        if len(v):
            qq=np.quantile(v,[.5,.95,.99]);result[label+'_median']=float(qq[0]);result[label+'_q95']=float(qq[1]);result[label+'_q99']=float(qq[2])
            result[label+'_mean_pooled']=float(v.mean())
        else:
            for suffix in ('median','q95','q99','mean_pooled'):result[label+'_'+suffix]=''
    result['row_status']='EMPTY' if not result['n_expected_queries'] else 'NO_SUBGROUP_OBSERVATIONS' if not result['n_subgroup_queries'] else 'OK'
    result['seed_weighting']='six seeds equal for equal_cells means/rates; pooled query counts/quantiles separately labelled'
    return result

def summaries():
    plan=load_plan();seed_rows=[];strata=[]
    for dataset in DATASETS:
        bundles=[tensors(dataset,s) for s in SEEDS]
        coarse=[(t,c,h,'all','all','all',sub) for t in TREATMENTS for c in ('all','same','cross') for h in ('all','same','different') for sub in ('all','correct','error')]
        coarse.append(('all','all','all','all','all','all','all'))
        fine=list(itertools.product(TREATMENTS,('same','cross'),('same','different'),SUFFIXES,DELAYS,QUERY_TYPES,('all','correct','error')))
        for definitions,destination in [(coarse,seed_rows),(fine,strata)]:
            for ti,(t,c,h,s,d,k,sub) in enumerate(definitions):
                selector=selectors(plan,c,h,s,d,t,k)
                metadata={'dataset':dataset,'treatment':t,'context':c,'history':h,'S':s,
                    'effective_suffix':('all' if s=='all' or t=='all' else s if t=='original' else SWAP[s]),
                    'd':d,'query_type':k,'donor_subset':sub}
                values=[]
                for seed,bundle in zip(SEEDS,bundles):
                    f,dd,a=selected(bundle,selector);r=measure(f,dd,a,sub);values.append(r)
                    destination.append({**metadata,'seed':seed,**r})
                merged=merge_seed_rows(values,bundles,selector,sub)
                destination.append({**metadata,'seed':'equal_weight',**merged})
                if ti%50==0:print(json.dumps({'phase':'summaries','dataset':dataset,'table':'seed' if destination is seed_rows else 'strata','row_group':ti}),flush=True)
        del bundles
    csv_write(OUT/'seed_summary.csv',seed_rows);csv_write(OUT/'strata_summary.csv',strata)

def bootstrap():
    plan=load_plan();result=[]
    for dataset in DATASETS:
        meta=plan['datasets'][dataset];orbits=len(meta['orbit_ids']);rng=np.random.default_rng(20261013)
        draws=rng.integers(orbits,size=(10000,orbits))
        weights=np.zeros((10000,orbits),np.int16)
        np.add.at(weights,(np.repeat(np.arange(10000),orbits),draws.ravel()),1)
        assert np.all(weights.sum(-1)==orbits)
        blocks=[dict(np.load(OUT/'data'/dataset/f'seed_{s}/orbit_blocks.npz')) for s in SEEDS]
        samples=[];names=[]
        for t in (*TREATMENTS,'all'):
            for c in ('all','same','cross'):
                for h in ('all','same','different'):
                    cells,ts,pairs,kinds=selectors(plan,c,h,'all','all',t,'all')
                    if not pairs:
                        result.append({'dataset':dataset,'treatment':t,'context':c,'history':h,'status':'EMPTY','95ci':None});continue
                    rr=[];balanced=True
                    for b in blocks:
                        ix=np.ix_(cells,ts,pairs,kinds,np.arange(orbits))
                        nn=b['agree'][ix];dd=b['counts'][ix]
                        flatden=dd.reshape(-1,orbits)
                        if np.all(flatden==flatden[:1]):
                            n=nn.sum(axis=(0,1,2,3));den=dd.sum(axis=(0,1,2,3))
                            rate_array=(weights@n)/(weights@den)
                        else:
                            balanced=False;parts=[]
                            fn=nn.reshape(-1,orbits).T;fd=flatden.T
                            for start in range(0,10000,128):
                                num=weights[start:start+128]@fn;den=weights[start:start+128]@fd
                                rates=np.divide(num,den,out=np.full(num.shape,np.nan),where=den>0)
                                parts.append(np.nanmean(rates,axis=1))
                            rate_array=np.concatenate(parts)
                        rr.append(rate_array)
                    mean=np.mean(rr,axis=0);names.append([t,c,h]);samples.append(mean)
                    result.append({'dataset':dataset,'treatment':t,'context':c,'history':h,'status':'OK',
                        'orbits':orbits,'replicates':10000,'rng_seed':20261013,'same_orbit_draws_all_6_seeds':True,
                        'all_directions_treatments_d_S_queries_retained':True,'95ci':np.quantile(mean,[.025,.975]).tolist(),
                        'balanced_cell_fast_reduction_verified':balanced})
        np.savez_compressed(OUT/f'block_bootstrap_{dataset}.npz',weights=weights,samples=np.array(samples),group_keys=np.array(names),orbit_ids=np.array(meta['orbit_ids']))
        print(json.dumps({'phase':'fixed model orbit bootstrap','dataset':dataset,'replicates':10000}),flush=True)
    write_json(OUT/'block_intervals.json',{'role':'conditional empirical interval for six fixed trained networks; not independent confirmation, no training or population uncertainty',
        'seed_resampling':False,'rng_seed':20261013,'replicates':10000,'intervals':result})

if __name__=='__main__':summaries();bootstrap()
