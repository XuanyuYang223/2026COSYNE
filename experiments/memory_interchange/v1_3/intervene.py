import itertools
import json
import numpy as np
from ..common import write_json
from ..intervene import continue_pairs,answers_from_content,correct_logp
from ..predict import save_csv
from ..truth import CONDITIONS,matrix_apply,mismatched_indices,inverse
from .controls import baseline_controls

MODES=('all','rec','both','three')

def frozen_masks(native_correct,mismatch):
    rec=np.broadcast_to(native_correct[None],(18,*native_correct.shape))
    both=rec & native_correct[:,None]
    three=both & native_correct[:,mismatch][:,None]
    return {'all':np.ones_like(rec,dtype=bool),'rec':rec,'both':both,'three':three}

def initial_counterfactual(cache,di,suffix):
    return np.array([[matrix_apply(matrix_apply(p,h,r),suffix,r) for r,h in CONDITIONS]
                      for p in cache['initial'][di]])

def quality(a):return np.where(a>=.95,2,np.where(a>=.90,1,0)).astype(np.int8)

def summarize(values,mask):
    count=int(mask.sum())
    return {'n':count,**{k:float(v[mask].mean()) if count else None for k,v in values.items()}}

def evaluate(model,cache,cfg,out,seed,exploratory=False):
    out.mkdir(parents=True,exist_ok=True)
    baseline_controls(model,cache,cfg,out)
    q,ids=cache['content'],cache['orbit_id'];n=cfg['n'];mismatch=mismatched_indices(ids,cfg['analysis']['mismatch_seed'])
    if exploratory:
        from .common import BASE
        legacy=np.load(BASE/'runs/pilot_128/analysis/mismatch.npz')
        assert np.array_equal(mismatch,legacy['indices'])
    np.savez_compressed(out/'mismatch.npz',indices=mismatch,recipient=q,donor=q[mismatch],orbit_id=ids)
    labels=[];directions=[];native_rows=[];mapping_rows=[];loss_blocks=[];count_blocks=[];advantage_blocks=[];ds=[];ss=[]
    donors,receivers=np.indices((18,18));orbits=np.unique(ids)
    for di,d in enumerate(cfg['timing']['delays']):
        for suffix in cfg['suffixes']:
            baseline=np.load(out/f'baseline_d{d}_S{suffix}.npz');native=baseline['native_logp']
            current,answers=answers_from_content(q,suffix,n)
            wrong_content,wrong_answers=answers_from_content(q[mismatch],suffix,n)
            correct=native.argmax(-1)==answers
            masks=frozen_masks(correct,mismatch)
            native_ce=-correct_logp(native,answers)
            acc=correct.mean(axis=(1,3));receiver_quality=quality(acc)
            native_pair_acc=(correct.mean(axis=(1,2,3))[:,None]+correct.mean(axis=(1,2,3))[None,:])/2
            pair_quality=quality(native_pair_acc)
            # Commit masks and quality strata before looking at any patch output.
            mask_path=out/f'frozen_masks_d{d}_S{suffix}.npz'
            np.savez_compressed(mask_path,**masks,native_correct=correct,native_answers=answers,
                                mismatch=mismatch,receiver_quality=receiver_quality,pair_quality=pair_quality,
                                content=q,orbit_id=ids)
            for a,(r,h) in enumerate(CONDITIONS):
                for kind in (0,1):
                    native_rows.append({'seed':seed,'condition':a,'r':r,'H':h,'S':suffix,'word':h+suffix,
                                        'non_N':sum(c!='N' for c in h+suffix),'d':d,'query_type':kind,
                                        'n':len(q)*n,'accuracy':float(correct[a,:,kind].mean()),
                                        'CE':float(native_ce[a,:,kind].mean()),'quality':int(receiver_quality[a,kind])})
            matched=continue_pairs(model,cache['memory'][di],cache['control'][di],donors.ravel(),receivers.ravel(),suffix,cfg).reshape(18,18,len(q),2,n,n)
            wrong=continue_pairs(model,cache['memory'][di][:,mismatch],cache['control'][di],donors.ravel(),receivers.ravel(),suffix,cfg).reshape(matched.shape)
            prediction=matched.argmax(-1);wrong_prediction=wrong.argmax(-1)
            signed=native_ce[None] + correct_logp(matched,answers[None])
            # ell_swap-ell_native = native logp - swap logp.
            signed=-signed
            wrong_signed=-native_ce[None]-correct_logp(wrong,answers[None])
            patch_correct=prediction==answers[None];wrong_correct=wrong_prediction==answers[None]
            cf_maps=initial_counterfactual(cache,di,suffix)
            cf_answers=np.stack((cf_maps,inverse(cf_maps)),axis=3)
            initial_agree=prediction==cf_answers
            initial_diff=cf_answers!=answers[None]
            wrong_diff=wrong_answers!=answers
            values={'matched_L':np.maximum(0,signed),'matched_R':correct[None]&~patch_correct,
                    'wrong_L':np.maximum(0,wrong_signed),'wrong_R':correct[None]&~wrong_correct,
                    'native_accuracy':np.broadcast_to(correct[None],masks['all'].shape),
                    'matched_accuracy':patch_correct,'wrong_accuracy':wrong_correct,
                    'matched_signed_CE':signed,'wrong_signed_CE':wrong_signed,
                    'repair':~correct[None]&patch_correct,'wrong_repair':~correct[None]&wrong_correct,
                    'native_CE':np.broadcast_to(native_ce[None],masks['all'].shape),
                    'matched_CE':-correct_logp(matched,answers[None]),'wrong_CE':-correct_logp(wrong,answers[None]),
                    'initial_cf_agree':initial_agree,'initial_cf_answer_differs':initial_diff,
                    'wrong_cf_agree':wrong_prediction==wrong_answers[None]}
            block_sums=np.zeros((18,18,len(orbits),len(MODES)))
            block_counts=np.zeros_like(block_sums);block_advantage=np.zeros_like(block_sums)
            for oi,oid in enumerate(orbits):
                keep=ids==oid
                for mi,mode in enumerate(MODES):
                    mask=masks[mode][:,:,keep]
                    block_sums[:,:,oi,mi]=(values['matched_L'][:,:,keep]*mask).sum((2,3,4))
                    block_counts[:,:,oi,mi]=mask.sum((2,3,4))
                    block_advantage[:,:,oi,mi]=((patch_correct[:,:,keep].astype(float)-wrong_correct[:,:,keep])*mask).sum((2,3,4))
            loss_blocks.append(block_sums);count_blocks.append(block_counts);advantage_blocks.append(block_advantage);ds.append(d);ss.append(suffix)
            for a,b in itertools.combinations(range(18),2):
                row={'seed':seed,'a':a,'b':b,'d':d,'S':suffix,'cross_context':int(CONDITIONS[a][0]!=CONDITIONS[b][0]),
                     'unseen_pair':int(CONDITIONS[a][1] in ('AB','BA') or CONDITIONS[b][1] in ('AB','BA')),
                     'native_ce':float((native_ce[a].mean()+native_ce[b].mean())/2),
                     'native_correct':float(native_pair_acc[a,b]),'quality':int(pair_quality[a,b])}
                pair_values={k:np.stack((v[a,b],v[b,a])) for k,v in values.items()}
                for mode in MODES:
                    mask=np.stack((masks[mode][a,b],masks[mode][b,a]));stats=summarize(pair_values,mask)
                    row.update({f'{k}_{mode}':v for k,v in stats.items()})
                    row[f'coverage_{mode}']=stats['n']/mask.size
                row.update(loss_positive=row['matched_L_all'],new_error=row['matched_R_all'],swap_correct=row['matched_accuracy_all'],
                           wrong_correct=row['wrong_accuracy_all'],wrong_cf_correct=row['wrong_cf_agree_all'])
                labels.append(row)
            for donor in range(18):
                for receiver in range(18):
                    if donor==receiver:continue
                    for kind in (0,1):
                        v={k:array[donor,receiver,:,kind] for k,array in values.items()}
                        for mode in MODES:
                            mask=masks[mode][donor,receiver,:,kind]
                            direction={'seed':seed,'donor':donor,'receiver':receiver,'r_donor':CONDITIONS[donor][0],
                                       'r_receiver':CONDITIONS[receiver][0],'H_donor':CONDITIONS[donor][1],
                                       'H_receiver':CONDITIONS[receiver][1],'d':d,'S':suffix,'query_type':kind,'mask':mode,
                                       'coverage':float(mask.mean()),'receiver_quality':int(receiver_quality[receiver,kind]),**summarize(v,mask)}
                            distinct=mask & initial_diff[donor,receiver,:,kind]
                            direction['initial_distinct_n']=int(distinct.sum())
                            direction['current_agree_distinct']=float(patch_correct[donor,receiver,:,kind][distinct].mean()) if distinct.any() else None
                            direction['initial_agree_distinct']=float(initial_agree[donor,receiver,:,kind][distinct].mean()) if distinct.any() else None
                            diff_wrong=mask & wrong_diff[receiver,:,kind]
                            direction['wrong_distinct_n']=int(diff_wrong.sum())
                            direction['wrong_cf_agree_distinct']=float((wrong_prediction[donor,receiver,:,kind]==wrong_answers[receiver,:,kind])[diff_wrong].mean()) if diff_wrong.any() else None
                            direction['wrong_original_agree_distinct']=float(wrong_correct[donor,receiver,:,kind][diff_wrong].mean()) if diff_wrong.any() else None
                            directions.append(direction)
                    equal_maps=(cf_maps[donor,receiver]==current[receiver]).all(1)
                    forward=prediction[donor,receiver,:,0];backward=prediction[donor,receiver,:,1]
                    cur_match=(forward==current[receiver]).all(1)&(backward==inverse(current[receiver])).all(1)
                    init_match=(forward==cf_maps[donor,receiver]).all(1)&(backward==inverse(cf_maps[donor,receiver])).all(1)
                    mapping_rows.append({'seed':seed,'donor':donor,'receiver':receiver,'H_donor':CONDITIONS[donor][1],
                         'H_receiver':CONDITIONS[receiver][1],'cross_context':int(CONDITIONS[donor][0]!=CONDITIONS[receiver][0]),
                         'd':d,'S':suffix,'n':len(q),'coincident_mapping_n':int(equal_maps.sum()),
                         'current_full_agree':float(cur_match.mean()),'initial_full_agree':float(init_match.mean()),
                         'distinct_mapping_n':int((~equal_maps).sum()),
                         'current_full_agree_distinct':float(cur_match[~equal_maps].mean()) if (~equal_maps).any() else None,
                         'initial_full_agree_distinct':float(init_match[~equal_maps].mean()) if (~equal_maps).any() else None})
            np.savez_compressed(out/f'raw_d{d}_S{suffix}.npz',matched_logp=matched,wrong_logp=wrong,native_logp=native,
                content=q,orbit_id=ids,initial=cache['initial'][di],mismatch_index=mismatch,current_future=current,
                answers=answers,initial_counterfactual_mapping=cf_maps,initial_counterfactual_answers=cf_answers,
                initial_answer_differs=initial_diff,initial_mapping_coincides=(cf_maps==current[None]).all(-1),
                counterfactual_answers=wrong_answers,wrong_final_content=wrong_content,seed=seed,d=d,S=suffix)
            print(json.dumps({'intervention_seed':seed,'d':d,'S':suffix,'native':float(correct.mean()),
                              'matched':float(np.mean([row['swap_correct'] for row in labels[-153:]]))}),flush=True)
    for name,rows in [('labels.csv',labels),('directions.csv',directions),('native_strata.csv',native_rows),('mapping_diagnostics.csv',mapping_rows)]:save_csv(out/name,rows)
    np.savez_compressed(out/'orbit_blocks.npz',loss_sums=np.array(loss_blocks),mask_counts=np.array(count_blocks),accuracy_advantage_sums=np.array(advantage_blocks),
                        orbit_ids=orbits,modes=np.array(MODES),delays=np.array(ds),suffixes=np.array(ss),seed=seed)
    grouped=[]
    for r in (0,1):
        for d in (0,10):
            for suffix in cfg['suffixes']:
                for kind in (0,1):
                    g=[v for v in native_rows if v['r']==r and v['d']==d and v['S']==suffix and v['query_type']==kind]
                    count=sum(v['n'] for v in g)
                    grouped.append({'r':r,'d':d,'S':suffix,'query_type':kind,'n':count,
                                    'accuracy':sum(v['n']*v['accuracy'] for v in g)/count,'CE':sum(v['n']*v['CE'] for v in g)/count})
    save_csv(out/'native_original_main_strata.csv',grouped)
    result={'seed':seed,'exploratory':exploratory,'contents':len(q),'orbits':len(orbits),'labels':len(labels),
            'native_min_stratum_accuracy':min(r['accuracy'] for r in native_rows),
            'native_strata_ge95':all(r['accuracy']>=.95 for r in native_rows),
            'native_strata_passed':all(r['accuracy']>=.95 for r in grouped),
            'compatibility_ceiling':bool(all(r['new_error']<.005 for r in labels) and np.subtract(*np.quantile([r['loss_positive'] for r in labels],[.75,.25]))<.01),
            'quality_bins':{str(i):sum(r['quality']==i for r in labels) for i in (0,1,2)},
            'mode_means':{mode:{key:float(np.mean([r[f'{key}_{mode}'] for r in labels if r[f'{key}_{mode}'] is not None]))
                               for key in ('matched_L','matched_R','wrong_L','wrong_R','repair','matched_accuracy','wrong_accuracy','wrong_cf_agree')}
                          for mode in MODES},
            'coverage':{mode:float(np.mean([r[f'coverage_{mode}'] for r in labels])) for mode in MODES}}
    write_json(out/'intervention_summary.json',result)
    return result
