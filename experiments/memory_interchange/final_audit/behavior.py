"""Donor-rule counterfactuals and a preregistered future-cue translation."""
import json
import time
import numpy as np
from ..cache import cache_states
from ..train import load_model
from ..intervene import continue_pairs, correct_logp, answers_from_content
from ..v1_3.controls import natural_outputs
from ..truth import CONDITIONS, matrix_apply
from .common import *

def directory(cohort, seed):
    p = OUT/cohort/f'seed_{seed}'; p.mkdir(parents=True, exist_ok=True); return p

def native_source(cohort, seed, d, s):
    root = RUNS/f'seed_{seed}/analysis' if cohort=='B1' else directory(cohort,seed)
    return root/f'baseline_d{d}_S{s}.npz'

def freeze_masks(cohort, seed, cache):
    out = directory(cohort,seed); records = []
    if (out/'mask_freeze.json').exists():
        receipt=json.loads((out/'mask_freeze.json').read_text())
        for record in receipt['files']:assert sha256(OUT/record['path'])==record['sha256']
        return
    nq = len(cache['content'])
    shape = (18,18,nq,2,7)
    cross = np.array([[CONDITIONS[a][0] != CONDITIONS[b][0] for b in range(18)] for a in range(18)])
    for d in (0,10):
        for s in SUFFIXES:
            baseline = np.load(native_source(cohort,seed,d,s))
            flipped = np.load(native_source(cohort,seed,d,SWAP[s]))
            native = baseline['native_logp']; natural = baseline['natural_logp']
            assert np.array_equal(native.argmax(-1), natural.argmax(-1))
            _, ans = answers_from_content(cache['content'],s,7)
            nc = natural.argmax(-1)==ans
            rec = np.broadcast_to(nc[None],shape)
            both = rec & nc[:,None]
            donor_prediction = np.where(cross[:,:,None,None,None],
                                       flipped['natural_logp'].argmax(-1)[:,None],
                                       natural.argmax(-1)[:,None])
            sem = rec & (donor_prediction==ans[None])
            masks = {'all':np.ones(shape,bool),'rec':rec,'both':both,'E_sem':sem}
            if cohort=='B1':
                old = np.load(RUNS/f'seed_{seed}/analysis'/f'frozen_masks_d{d}_S{s}.npz')
                for k in ('all','rec','both'): assert np.array_equal(masks[k], old[k])
            p = out/f'masks_d{d}_S{s}.npz'
            np.savez_compressed(p,**masks,native_correct=nc,answers=ans,
                                content=cache['content'],orbit_id=cache['orbit_id'])
            records.append({'d':d,'S':s,'path':str(p.relative_to(OUT)), 'sha256':sha256(p),
                            'coverage':{k:float(v.mean()) for k,v in masks.items()}})
    write_json(out/'mask_freeze.json',{'time':time.time(),'patch_outputs_not_yet_computed':True,
            'natural_outputs_define_masks':True,'files':records})

def prepare_natives():
    if (OUT/'all_native_masks_frozen.json').exists():
        receipt=json.loads((OUT/'all_native_masks_frozen.json').read_text())
        for entry in receipt['seeds']:
            seed=entry['seed']
            assert sha256(directory('B_reserved',seed)/'mask_freeze.json')==entry['mask_receipt_sha256']
            assert sha256(directory('B1',seed)/'mask_freeze.json')==entry['B1_mask_receipt_sha256']
        return
    cfg = config(); seed_all(123,4,True)
    members = np.load(OUT/'B_reserved_members.npz')
    receipt = []
    for seed in SEEDS:
        out = directory('B_reserved',seed)
        if (out/'mask_freeze.json').exists():
            freeze_masks('B1',seed,dict(np.load(RUNS/f'seed_{seed}/analysis/behavior_states.npz')))
            receipt.append({'seed':seed,'mask_receipt_sha256':sha256(out/'mask_freeze.json'),
                            'B1_mask_receipt_sha256':sha256(directory('B1',seed)/'mask_freeze.json')}); continue
        model,_ = load_model(BASE/json.loads((RUNS/f'seed_{seed}/summary.json').read_text())['selected_checkpoint'])
        with Budget() as budget:
            cache = cache_states(model,members['content'],members['orbit_id'],cfg,out/'states.npz')
            rows = []
            for di,d in enumerate((0,10)):
                for s in SUFFIXES:
                    budget.check()
                    m,c = cache['memory'][di],cache['control'][di]
                    native = continue_pairs(model,m,c,range(18),range(18),s,cfg)
                    natural = natural_outputs(model,cache,di,s,cfg)
                    sham = continue_pairs(model,m.copy(),c.copy(),range(18),range(18),s,cfg)
                    np.testing.assert_allclose(native,natural,atol=1e-5,rtol=1e-5)
                    assert np.array_equal(native,sham)
                    assert np.array_equal(native.argmax(-1),natural.argmax(-1))
                    np.savez_compressed(out/f'baseline_d{d}_S{s}.npz',native_logp=native,
                                        natural_logp=natural,sham_logp=sham)
                    rows.append({'d':d,'S':s,'native_uninterrupted_max_logp_error':float(np.abs(native-natural).max()),
                                 'argmax_changes':0,'sham_max_logp_error':0.})
                    print(json.dumps({'phase':'native','seed':seed,'d':d,'S':s}),flush=True)
        write_json(out/'controls.json',{'GPU_registered_tolerance_passed':True,'atol':1e-5,'rtol':1e-5,'rows':rows})
        freeze_masks('B_reserved',seed,cache)
        b1cache = dict(np.load(RUNS/f'seed_{seed}/analysis/behavior_states.npz'))
        freeze_masks('B1',seed,b1cache)
        receipt.append({'seed':seed,'mask_receipt_sha256':sha256(out/'mask_freeze.json'),
                        'B1_mask_receipt_sha256':sha256(directory('B1',seed)/'mask_freeze.json')})
    write_json(OUT/'all_native_masks_frozen.json',{'time':time.time(),'seeds':receipt,
               'before_all_new_memory_patch_outputs':True})

def candidate_stats(logp, candidates, subsets, meta):
    pred = logp.argmax(-1); rows = []
    for kind in (-1,0,1):
        selector = slice(None) if kind==-1 else slice(kind,kind+1)
        p = pred[:,selector]
        for subset, mask in subsets.items():
            mask = mask[:,selector]; count = int(mask.sum())
            row = {**meta,'query_type':kind,'subset':subset,'n':count,'coverage':count/mask.size}
            for name in ('receiver','initial','donor_rule'):
                if name not in candidates:
                    row[name+'_agree']=''; row[name+'_CE']=''; continue
                a=candidates[name]
                a = a[:,selector]
                row[name+'_agree'] = float((p==a)[mask].mean()) if count else ''
                lp = correct_logp(logp[:,selector],a)
                row[name+'_CE'] = float(-lp[mask].astype(np.float64).mean()) if count else ''
            rows.append(row)
    return rows

def counterfactual_B1():
    for seed in SEEDS:
        out = directory('B1',seed)
        if (out/'counterfactual_summary.json').exists(): continue
        rows=[]; maps=[]
        for d in (0,10):
            for s in SUFFIXES:
                raw = dict(np.load(RUNS/f'seed_{seed}/analysis'/f'raw_d{d}_S{s}.npz'))
                shape = raw['matched_logp'].shape[:-1]
                rec = np.broadcast_to(raw['answers'][None],shape)
                donor = np.broadcast_to(raw['answers'][:,None],shape)
                initial = raw['initial_counterfactual_answers']
                # Independently reconstruct the initial-input interpretation.
                independent = np.array([[matrix_apply(matrix_apply(raw['initial'][a],CONDITIONS[b][1],CONDITIONS[b][0]),s,CONDITIONS[b][0])
                                         for b in range(18)] for a in range(18)])
                assert np.array_equal(independent,raw['initial_counterfactual_mapping'])
                assert np.array_equal(maps_answers(independent),initial)
                ri=rec!=initial; rd=rec!=donor; idiff=initial!=donor
                recw=np.broadcast_to(raw['counterfactual_answers'][None],shape)
                donw=np.broadcast_to(raw['counterfactual_answers'][:,None],shape)
                saved = {'matched_prediction':raw['matched_logp'].argmax(-1),
                         'wrong_prediction':raw['wrong_logp'].argmax(-1),
                         'Q_receiver':rec,'Q_initial':initial,'Q_donor_rule':donor,
                         'wrong_Q_receiver':recw,'wrong_Q_donor_rule':donw,
                         'receiver_initial_distinct':ri,'receiver_donor_distinct':rd,
                         'initial_donor_distinct':idiff,'all_three_distinct':ri&rd&idiff,
                         'wrong_rules_distinct':recw!=donw,
                         'content':raw['content'],'orbit_id':raw['orbit_id'],'mismatch_index':raw['mismatch_index'],
                         'initial':raw['initial'],'Q_receiver_mapping':rec[...,0,:],
                         'Q_initial_mapping':independent,'Q_donor_rule_mapping':donor[...,0,:]}
                np.savez_compressed(out/f'counterfactual_d{d}_S{s}.npz',**saved)
                for a,b in PAIRS:
                    meta={'seed':seed,'d':d,'S':s,'donor':a,'receiver':b,
                          'r_donor':CONDITIONS[a][0],'r_receiver':CONDITIONS[b][0],
                          'H_donor':CONDITIONS[a][1],'H_receiver':CONDITIONS[b][1],
                          'cross':int(CONDITIONS[a][0]!=CONDITIONS[b][0])}
                    candidates={'receiver':rec[a,b],'initial':initial[a,b],'donor_rule':donor[a,b]}
                    subsets={'all':np.ones_like(rd[a,b]),'receiver_initial_distinct':ri[a,b],
                        'receiver_donor_distinct':rd[a,b],'initial_donor_distinct':idiff[a,b],
                        'all_three_distinct':ri[a,b]&rd[a,b]&idiff[a,b]}
                    rows += candidate_stats(raw['matched_logp'][a,b],candidates,subsets,{**meta,'treatment':'matched'})
                    rows += candidate_stats(raw['wrong_logp'][a,b],{'receiver':recw[a,b],
                           'donor_rule':donw[a,b]}, {'all':np.ones_like(rd[a,b]),
                           'receiver_donor_distinct':recw[a,b]!=donw[a,b]}, {**meta,'treatment':'wrong'})
                    # Full mappings from all 7 item and all 7 position queries.
                    for name in ('matched','wrong'):
                        pred=raw[name+'_logp'][a,b].argmax(-1)
                        fwd,rev=pred[:,0],pred[:,1]
                        bij=np.all(np.sort(fwd,axis=-1)==np.arange(7),axis=-1)
                        revbij=np.all(np.sort(rev,axis=-1)==np.arange(7),axis=-1)
                        mutual=bij & revbij & np.all(np.take_along_axis(rev,fwd,axis=-1)==np.arange(7),axis=-1)
                        cs = candidates if name=='matched' else {'receiver':recw[a,b],'donor_rule':donw[a,b]}
                        md=np.any(cs['receiver']!=cs['donor_rule'],axis=(-1,-2))
                        msub={'all':np.ones(len(pred),bool),'receiver_donor_distinct':md}
                        if name=='matched':
                            mir=np.any(cs['receiver']!=cs['initial'],axis=(-1,-2))
                            mid=np.any(cs['donor_rule']!=cs['initial'],axis=(-1,-2))
                            msub.update(receiver_initial_distinct=mir,initial_donor_distinct=mid,
                                        all_three_distinct=md&mir&mid)
                        for sn,sm in msub.items():
                            count=int(sm.sum())
                            row={**meta,'treatment':name,'subset':sn,'n_mappings':count,'coverage':count/len(pred),
                                'item_bijection':float(bij[sm].mean()) if count else '',
                                'position_bijection':float(revbij[sm].mean()) if count else '',
                                'mutually_inverse':float(mutual[sm].mean()) if count else ''}
                            for cn in ('receiver','initial','donor_rule'):
                                row[cn+'_full_mapping_agree']=float(np.all(pred==cs[cn],axis=(-1,-2))[sm].mean()) if cn in cs and count else ''
                            maps.append(row)
                print(json.dumps({'phase':'B1 counterfactual','seed':seed,'d':d,'S':s}),flush=True)
        csv_write(out/'candidate_agreement.csv',rows);csv_write(out/'mapping_agreement.csv',maps)
        write_json(out/'counterfactual_summary.json',{'status':'EXPLORATORY','seed':seed,'query_rows':len(rows),
                   'mapping_rows':len(maps),'source':'all original stored matched/wrong log-probabilities; no content selection',
                   'wrong_initial_column':'blank: only receiver/donor rule candidates defined for wrong content'})

def analyze_translation(cohort,seed,d,s,cache,original,translated):
    out=directory(cohort,seed);rows=[];native_rows=[]
    masks=dict(np.load(out/f'masks_d{d}_S{s}.npz'))
    baseline=np.load(native_source(cohort,seed,d,s)); flip=np.load(native_source(cohort,seed,d,SWAP[s]))
    native=baseline['native_logp']; native_t=flip['native_logp']; sham=baseline['sham_logp']
    _, target=answers_from_content(cache['content'],s,7)
    _, literal=answers_from_content(cache['content'],SWAP[s],7)
    lp_native=correct_logp(native,target); nc=native.argmax(-1)==target
    for b,(r,h) in enumerate(CONDITIONS):
        for kind in (-1,0,1):
            sl=slice(None) if kind==-1 else slice(kind,kind+1)
            for name,output in [('native',native),('native_translated',native_t),('sham',sham)]:
                pred=output[b,:,sl].argmax(-1)
                native_rows.append({'seed':seed,'d':d,'S':s,'translated_S':SWAP[s],'receiver':b,'r':r,'H':h,
                    'query_type':kind,'treatment':name,'n':pred.size,
                    'original_target_accuracy':float((pred==target[b,:,sl]).mean()),
                    'literal_target_accuracy':float((pred==literal[b,:,sl]).mean()),
                    'original_target_CE':float(-correct_logp(output[b,:,sl],target[b,:,sl]).astype(np.float64).mean()),
                    'literal_target_CE':float(-correct_logp(output[b,:,sl],literal[b,:,sl]).astype(np.float64).mean())})
    metrics=('accuracy_original','CE_original','new_error_original','loss_positive_original',
             'accuracy_literal','CE_literal','receiver_rule_agree','donor_rule_agree')
    sums=[];counts=[];orbit_ids=np.unique(cache['orbit_id'])
    for a,b in PAIRS:
        for kind in (-1,0,1):
            sl=slice(None) if kind==-1 else slice(kind,kind+1)
            tt=target[b,:,sl];lt=literal[b,:,sl];orig=original[a,b,:,sl];trans=translated[a,b,:,sl]
            pp=[orig.argmax(-1),trans.argmax(-1)]
            donor_ans=[target[a,:,sl],literal[a,:,sl]]
            result=[]
            for output,p,da,rt in zip((orig,trans),pp,donor_ans,(tt,lt)):
                ce=-correct_logp(output,tt).astype(np.float64)
                values=np.stack([p==tt,ce,nc[b,:,sl] & (p!=tt),np.maximum(0,ce+lp_native[b,:,sl]),
                          p==lt,-correct_logp(output,lt).astype(np.float64),p==rt,p==da],axis=-1).astype(np.float64)
                result.append(values)
            for mode in MODES:
                mask=masks[mode][a,b,:,sl]; n=int(mask.sum())
                row={'seed':seed,'cohort':cohort,'d':d,'S':s,'translated_S':SWAP[s],
                     'donor':a,'receiver':b,'r_donor':CONDITIONS[a][0],'r_receiver':CONDITIONS[b][0],
                     'H_donor':CONDITIONS[a][1],'H_receiver':CONDITIONS[b][1],
                     'cross':int(CONDITIONS[a][0]!=CONDITIONS[b][0]),'query_type':kind,'mode':mode,'n':n,
                     'total_n':mask.size,'coverage':n/mask.size}
                distinct=mask & (tt!=donor_ans[0])
                row['rule_distinct_n']=int(distinct.sum())
                for ti,tname in enumerate(('original','translated')):
                    row[tname+'_donor_rule_agree_distinct']=float(result[ti][...,7][distinct].mean()) if distinct.any() else ''
                    row[tname+'_receiver_rule_agree_distinct']=float(result[ti][...,6][distinct].mean()) if distinct.any() else ''
                for mi,name in enumerate(metrics):
                    for ti,tname in enumerate(('original','translated')):
                        row[tname+'_'+name]=float(result[ti][...,mi][mask].mean()) if n else ''
                    row['delta_'+name]=(row['translated_'+name]-row['original_'+name]) if n else ''
                rows.append(row)
            if kind==-1:
                # Shape direction, mode, treatment, orbit, metric; same paired mask.
                sums.append(np.array([[[v[cache['orbit_id']==orb][masks[mode][a,b][cache['orbit_id']==orb]].sum(axis=0)
                                       for orb in orbit_ids] for v in result] for mode in MODES]))
                counts.append(np.array([[masks[mode][a,b][cache['orbit_id']==orb].sum() for orb in orbit_ids]
                                        for mode in MODES]))
    np.savez_compressed(out/f'blocks_d{d}_S{s}.npz',sums=np.array(sums),counts=np.array(counts),
        orbit_ids=orbit_ids,metrics=np.array(metrics),modes=np.array(MODES),directions=np.array(PAIRS))
    csv_write(out/f'translation_d{d}_S{s}.csv',rows)
    csv_write(out/f'native_d{d}_S{s}.csv',native_rows)

def run_patches():
    assert (OUT/'all_native_masks_frozen.json').exists()
    cfg=config();seed_all(123,4,True)
    don,rec=np.indices((18,18))
    for cohort in ('B1','B_reserved'):
        for seed in SEEDS:
            out=directory(cohort,seed)
            if (out/'translation_complete.json').exists():continue
            cache=dict(np.load(RUNS/f'seed_{seed}/analysis/behavior_states.npz' if cohort=='B1' else out/'states.npz'))
            model,_=load_model(BASE/json.loads((RUNS/f'seed_{seed}/summary.json').read_text())['selected_checkpoint'])
            controls=[]
            for di,d in enumerate((0,10)):
                for s in SUFFIXES:
                    budget=Budget()
                    with budget:
                        m,c=cache['memory'][di],cache['control'][di]
                        if cohort=='B1':
                            original=np.load(RUNS/f'seed_{seed}/analysis'/f'raw_d{d}_S{s}.npz')['matched_logp']
                        else:
                            original=continue_pairs(model,m,c,don.ravel(),rec.ravel(),s,cfg).reshape(18,18,len(cache['content']),2,7,7)
                        translated=continue_pairs(model,m,c,don.ravel(),rec.ravel(),SWAP[s],cfg).reshape(original.shape)
                    if s=='NN':assert np.array_equal(original,translated)
                    olddiff=0.
                    if cohort=='B1':
                        expected=np.load(RUNS/f'seed_{seed}/analysis'/f'raw_d{d}_S{SWAP[s]}.npz')['matched_logp']
                        olddiff=float(np.abs(expected-translated).max())
                        np.testing.assert_allclose(expected,translated,atol=1e-5,rtol=1e-5)
                    _,ans=answers_from_content(cache['content'],s,7)
                    _,lit=answers_from_content(cache['content'],SWAP[s],7)
                    np.savez_compressed(out/f'translation_d{d}_S{s}.npz',
                        patch_original_logp=original,patch_translated_logp=translated,
                        Q_original=ans,Q_literal=lit,donor_rule_original=ans,donor_rule_translated=lit,
                        patch_original_mapping_predictions=original.argmax(-1),
                        patch_translated_mapping_predictions=translated.argmax(-1),
                        content=cache['content'],orbit_id=cache['orbit_id'],d=d,S=s,translated_S=SWAP[s],seed=seed)
                    analyze_translation(cohort,seed,d,s,cache,original,translated)
                    controls.append({'d':d,'S':s,'NN_max_logp_difference':0. if s=='NN' else None,
                                      'stored_B1_translated_source_max_difference':olddiff if cohort=='B1' else None})
                    print(json.dumps({'phase':'translation','cohort':cohort,'seed':seed,'d':d,'S':s}),flush=True)
            write_json(out/'translation_complete.json',{'seed':seed,'cohort':cohort,'controls':controls,
                'mask_receipt_sha256':sha256(out/'mask_freeze.json'),'completed_time':time.time()})

def summarize_translation():
    summaries=[]
    for cohort in ('B1','B_reserved'):
        for seed in SEEDS:
            out=directory(cohort,seed)
            rows=[r for d in (0,10) for s in SUFFIXES for r in csv_read(out/f'translation_d{d}_S{s}.csv') if r['query_type']=='-1']
            numeric=[k for k in rows[0] if k.startswith(('original_','translated_','delta_')) and k!='translated_S']
            for cross in (0,1):
                for mode in MODES:
                    for suffix in ('ALL',*SUFFIXES):
                        for delay in ('ALL',0,10):
                            selected=[r for r in rows if int(r['cross'])==cross and r['mode']==mode and
                                      (suffix=='ALL' or r['S']==suffix) and (delay=='ALL' or int(r['d'])==delay)]
                            row={'cohort':cohort,'seed':seed,'cross':cross,'mode':mode,'S':suffix,'d':delay,
                                 'direction_cells':len(selected),'query_n':sum(int(r['n']) for r in selected),
                                 'coverage':sum(int(r['n']) for r in selected)/sum(int(r['total_n']) for r in selected)}
                            row.update({k:float(np.mean([float(r[k]) for r in selected if r[k]!=''])) if any(r[k]!='' for r in selected) else '' for k in numeric})
                            summaries.append(row)
    csv_write(OUT/'translation_seed_summary.csv',summaries)
    overall=[]
    for cohort in ('B1','B_reserved'):
        for cross in (0,1):
            for mode in MODES:
                for s in ('ALL',*SUFFIXES):
                    for d in ('ALL','0','10'):
                        selected=[r for r in summaries if r['cohort']==cohort and r['cross']==cross and r['mode']==mode and r['S']==s and str(r['d'])==d]
                        overall.append({'cohort':cohort,'cross':cross,'mode':mode,'S':s,'d':d,
                            'seeds':len(selected),'coverage':float(np.mean([r['coverage'] for r in selected])),
                            **{k:float(np.mean([r[k] for r in selected if r[k]!=''])) if any(r[k]!='' for r in selected) else '' for k in numeric}})
    csv_write(OUT/'translation_overall_summary.csv',overall)
    # Paired seed/orbit bootstrap of the newly frozen primary contrast. No fit.
    arrays={}
    for seed in SEEDS:
        blocks=[dict(np.load(directory('B_reserved',seed)/f'blocks_d{d}_S{s}.npz')) for d in (0,10) for s in SUFFIXES]
        cross=np.array([CONDITIONS[a][0]!=CONDITIONS[b][0] for a,b in PAIRS])
        num=np.concatenate([b['sums'][cross] for b in blocks],axis=0)
        den=np.concatenate([b['counts'][cross] for b in blocks],axis=0)
        arrays[seed]=(num,den)
    rng=np.random.default_rng(20261018);samples=[]
    for rep in range(2000):
        selected=rng.choice(SEEDS,6,replace=True); weights={}
        for seed in set(selected): weights[seed]=np.bincount(rng.integers(51,size=51),minlength=51)
        values=[]
        for seed in selected:
            num,den=arrays[seed];w=weights[seed]
            nn=np.einsum('cmton,o->cmtn',num,w);dd=np.einsum('cmo,o->cm',den,w)
            v=np.divide(nn[:,:,1,0]-nn[:,:,0,0],dd,out=np.full_like(dd,np.nan,dtype=float),where=dd>0)
            values.append(np.nanmean(v,axis=0))
        samples.append(np.mean(values,axis=0))
    np.savez_compressed(OUT/'translation_block_bootstrap.npz',samples=np.array(samples),modes=np.array(MODES),seed=20261018)
    write_json(OUT/'translation_block_intervals.json',{'estimand':'cross-context patch translation minus original, original semantic target, all 3 suffixes',
        'models_and_masks_fixed':True,'paired_seed_whole_orbit_resampling':True,'replicates':2000,
        'intervals':{m:np.quantile(np.array(samples)[:,i],[.025,.975]).tolist() for i,m in enumerate(MODES)}})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['native','CF','patch','summary']);a=p.parse_args()
    {'native':prepare_natives,'CF':counterfactual_B1,'patch':run_patches,'summary':summarize_translation}[a.phase]()
