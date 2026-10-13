"""Compare existing outputs by semantic keys; never execute a network."""
import time
import numpy as np
from ..truth import CONDITIONS, matrix_apply, inverse
from .common import *

def bool_flag(bits,name):return (bits & FLAG[name])!=0

def maps_answers(q):return np.stack((q,inverse(q)),axis=1)

def compare(pp,dd,pa,da,truth,available,pvalid,ptie,dtie,ptie_known,dtie_known):
    flags=np.zeros(pa.shape,np.uint16)
    flags[available]|=FLAG['available']
    flags[available & (pa==da)]|=FLAG['agree']
    flags[available & (da==truth)]|=FLAG['donor_correct']
    flags[ptie_known & available]|=FLAG['patch_tie_known']
    flags[dtie_known & available]|=FLAG['donor_tie_known']
    flags[ptie & available]|=FLAG['patch_tie']
    flags[dtie & available]|=FLAG['donor_tie']
    valid=available & pvalid
    flags[valid]|=FLAG['probability_available']
    differences=np.full((*pa.shape,2),np.nan,np.float64)
    elements=np.full(pa.shape,255,np.uint8)
    if pp is not None and dd is not None:
        absdiff=np.abs(pp-dd);maximum=absdiff.max(-1);tv=.5*absdiff.sum(-1)
        differences[...,0][valid]=maximum[valid];differences[...,1][valid]=tv[valid]
        equal=(pp==dd).sum(-1).astype(np.uint8);elements[valid]=equal[valid]
        flags[valid & (equal==7)]|=FLAG['exact']
        for tolerance,name in [(1e-6,'close_1e6'),(1e-5,'close_1e5'),(1e-4,'close_1e4')]:
            flags[valid & (maximum<=tolerance)]|=FLAG[name]
    answers=np.stack((pa,da,truth,elements),axis=-1).astype(np.uint8)
    return flags,differences,answers

def output_arrays(path,field):
    f=np.load(path);value=f[field]
    if field.endswith('logp'):
        p,pvalid,diagnostic=probabilities(value)
        finite=np.isfinite(value).all(-1)
        answer=value.argmax(-1).astype(np.uint8)
        tie=(value==value.max(-1,keepdims=True)).sum(-1)>1
        if pvalid.any():assert np.array_equal(answer[pvalid],p.argmax(-1)[pvalid])
        return p,answer,finite,pvalid,tie,finite,diagnostic
    # The fallback is an actual stored prediction, never an answer/target array.
    assert field.endswith('mapping_predictions')
    answer=value.astype(np.uint8);finite=(value>=0)&(value<7)
    return None,answer,finite,np.zeros(value.shape,bool),np.zeros(value.shape,bool),np.zeros(value.shape,bool),{'answer_only':True}

def score_control(logp,reference,name,meta):
    p,v,diag=probabilities(logp);r,vr,_=probabilities(reference);ok=v&vr
    diff=np.abs(p-r);agree=logp.argmax(-1)==reference.argmax(-1)
    return {**meta,'control':name,'n':int(ok.sum()),'answer_agree_n':int((ok&agree).sum()),
        'answer_agreement':float(agree[ok].mean()) if ok.any() else None,
        'max_abs_probability_difference':float(diff.max(-1)[ok].max()) if ok.any() else None,
        'TV_mean':float((.5*diff.sum(-1))[ok].mean()) if ok.any() else None,
        'exact_probability_query_n':int((ok & (p==r).all(-1)).sum()),'different_external_input':False}

def run():
    plan=load_plan();start=time.perf_counter()
    for path,rec in plan['input_files'].items():
        if rec['exists']:assert sha(BASE/path)==rec['sha256'],path
    FLAG.update(patch_tie_known=1024,donor_tie_known=2048)
    missing=[];pairing=[];control_rows=[];c_rows=[];norm_rows=[];nn_rows=[];native_seen=set();detailed_count=0
    index_entries=[]
    for dataset in DATASETS:
        meta=dataset_meta(plan,dataset);q=np.array(meta['members']);ids=np.array(meta['orbit_ids_per_content'])
        unique=np.array(meta['orbit_ids']);nq=len(q)
        for seed in SEEDS:
            out=OUT/'data'/dataset/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            shape=(6,2,306,nq,2,7)
            flags=np.lib.format.open_memmap(out/'flags.npy',mode='w+',dtype=np.uint16,shape=shape);flags[:]=0
            diff=np.lib.format.open_memmap(out/'probability_differences.npy',mode='w+',dtype=np.float64,shape=(*shape,2));diff[:]=np.nan
            answers=np.lib.format.open_memmap(out/'answers.npy',mode='w+',dtype=np.uint8,shape=(*shape,4));answers[:]=255
            block_agree=np.zeros((6,2,306,2,len(unique)),np.int32);block_n=np.zeros_like(block_agree)
            sm=next((m for m in plan['state_metadata'] if m['dataset']==dataset and m['seed']==seed),None)
            cached_c=None
            if sm and sm['valid']:
                cached_c=np.load(BASE/sm['path'])['control']
                for di,d in enumerate(DELAYS):
                    for a in range(18):
                        for b in range(18):
                            ca=cached_c[di,a].astype(np.float64);cb=cached_c[di,b].astype(np.float64)
                            c_rows.append({'dataset':dataset,'seed':seed,'d':d,'donor_condition':condition_key(*CONDITIONS[a]),
                                'recipient_condition':condition_key(*CONDITIONS[b]),'donor_id':a,'recipient_id':b,
                                'cross_context':int(CONDITIONS[a][0]!=CONDITIONS[b][0]),
                                'same_history':int(CONDITIONS[a][1]==CONDITIONS[b][1]),'is_diagonal':int(a==b),
                                'c_exact_equal':bool(np.array_equal(cached_c[di,a],cached_c[di,b])),
                                'raw_c_L2':float(np.linalg.norm(ca-cb)),'raw_c_max_abs':float(np.abs(ca-cb).max())})
            entries=[e for e in plan['entries'] if e['dataset']==dataset and e['seed']==seed]
            for entry in entries:
                d=entry['d'];s=entry['original_S'];ti=TREATMENTS.index(entry['treatment']);cell=DELAYS.index(d)*3+SUFFIXES.index(s)
                if not entry['available']:
                    missing.append({**entry,'reason':'prediction missing; no neural replay or truth substitution'})
                    pairing.append({k:entry[k] for k in ('dataset','seed','d','original_S','treatment','effective_suffix') }|{'expected':entry['expected_queries_nonself'],'paired':0,'missing':entry['expected_queries_nonself']})
                    continue
                patchfile=BASE/entry['patch_path'];nativefile=BASE/entry['native_path']
                pp_all,pa_all,pfa_all,pva_all,pt_all,ptk_all,pdiag=output_arrays(patchfile,entry['patch_field'])
                dd_all,da_all,dfa_all,dva_all,dt_all,dtk_all,ddiag=output_arrays(nativefile,entry['native_field'])
                pidx=np.array(entry['patch_q_index']);nidx=np.array(entry['native_q_index']);qvalid=(pidx>=0)&(nidx>=0)
                psafe=np.maximum(pidx,0);nsafe=np.maximum(nidx,0)
                a,b=PAIRS.T
                def pselect(x):return x[a,b][:,psafe] if x is not None else None
                def dselect(x):return x[a][:,nsafe] if x is not None else None
                pp=pselect(pp_all);dd=dselect(dd_all);pa=pselect(pa_all);da=dselect(da_all)
                available=pselect(pfa_all)&dselect(dfa_all)&qvalid[None,:,None,None]
                probvalid=pselect(pva_all)&dselect(dva_all)&qvalid[None,:,None,None]
                truth=np.stack([maps_answers(matrix_apply(q,entry['effective_suffix'],r)) for r,h in CONDITIONS])[a]
                fl,delta,pred=compare(pp,dd,pa,da,truth,available,probvalid,pselect(pt_all),dselect(dt_all),pselect(ptk_all),dselect(dtk_all))
                flags[cell,ti]=fl;diff[cell,ti]=delta;answers[cell,ti]=pred
                for oi,orb in enumerate(unique):
                    keep=ids==orb
                    block_n[cell,ti,:,:,oi]=available[:,keep].sum(axis=(1,3))
                    block_agree[cell,ti,:,:,oi]=((pa==da)&available)[:,keep].sum(axis=(1,3))
                n=int(available.sum());expected=entry['expected_queries_nonself']
                pairing.append({k:entry[k] for k in ('dataset','seed','d','original_S','treatment','effective_suffix')}|
                    {'expected':expected,'paired':n,'missing':expected-n,'probability_queries':int(probvalid.sum()),
                     'unique_patch_keys':n,'native_matches_per_patch_key':1,'accidental_many_to_many':False})
                index_entries.append({**entry,'tensor_path':str(out.relative_to(OUT)),'tensor_cell':cell,'treatment_index':ti,
                    'patch_donor_indices':a.tolist(),'patch_recipient_indices':b.tolist(),
                    'native_condition_indices':a.tolist(),'canonical_content_ids':meta['content_ids'],
                    'source_probabilities_stored_once_in_original_input_files':True})
                norm_rows.append({'dataset':dataset,'seed':seed,'d':d,'S':s,'treatment':entry['treatment'],
                    'source':'patch','field':entry['patch_field'],**pdiag})
                nativekey=(str(nativefile),entry['native_field'])
                if nativekey not in native_seen:
                    norm_rows.append({'dataset':dataset,'seed':seed,'d':d,'S':entry['effective_suffix'],'treatment':'native',
                        'source':'native','field':entry['native_field'],**ddiag})
                    baseline=np.load(nativefile)
                    for name,field in [('sham','sham_logp'),('uninterrupted','natural_logp')]:
                        if field in baseline:
                            control_rows.append(score_control(baseline[field],baseline[entry['native_field']],name,
                                {'dataset':dataset,'seed':seed,'d':d,'effective_suffix':entry['effective_suffix']}))
                    if 'full_logp' in baseline:
                        donors=baseline['full_donor'];receivers=baseline['full_receiver']
                        assert len(donors)==len(receivers) and len(set(zip(map(int,donors),map(int,receivers))))==len(donors)
                        assert all(CONDITIONS[a0][0]==CONDITIONS[b0][0] for a0,b0 in zip(donors,receivers))
                        control_rows.append(score_control(baseline['full_logp'],baseline[entry['native_field']][donors],
                            'full_state_same_future_input',{'dataset':dataset,'seed':seed,'d':d,'effective_suffix':entry['effective_suffix']}))
                    native_seen.add(nativekey)
                # All available diagonal records remain implementation controls.
                if pp_all is not None:
                    diag=pp_all[np.arange(18),np.arange(18)][:,psafe]
                    dn=dd_all[:,nsafe];diagvalid=pva_all[np.arange(18),np.arange(18)][:,psafe]&dva_all[:,nsafe]&qvalid[None,:,None,None]
                    de=np.abs(diag-dn)
                    control_rows.append({'dataset':dataset,'seed':seed,'d':d,'effective_suffix':entry['effective_suffix'],
                        'S':s,'treatment':entry['treatment'],'control':'patch_diagonal','n':int(diagvalid.sum()),
                        'answer_agree_n':int(((diag.argmax(-1)==dn.argmax(-1))&diagvalid).sum()),
                        'answer_agreement':float((diag.argmax(-1)==dn.argmax(-1))[diagvalid].mean()),
                        'max_abs_probability_difference':float(de.max(-1)[diagvalid].max()),
                        'TV_mean':float((.5*de.sum(-1))[diagvalid].mean()),
                        'exact_probability_query_n':int(((diag==dn).all(-1)&diagvalid).sum()),'different_external_input':False})
                if s=='NN' and ti==1:
                    pf=np.load(patchfile)
                    origfield='patch_original_logp' if 'patch_original_logp' in pf else 'patch_original_mapping_predictions'
                    assert np.array_equal(pf[origfield],pf[entry['patch_field']])
                    assert np.array_equal(flags[cell,0],flags[cell,1])
                    assert np.array_equal(diff[cell,0],diff[cell,1],equal_nan=True)
                    assert native_path(dataset,seed,d,s)==native_path(dataset,seed,d,SWAP[s])
                    nn_rows.append({'dataset':dataset,'seed':seed,'d':d,'same_native_source':True,
                        'all_324_patch_directions_identical':True,'all_main_comparisons_identical':True})
                # Explicit full probability records for answer differences and
                # registered numerical non-closeness. All smaller differences
                # remain in compact tensors + lossless source index.
                different=available & ((pa!=da) | (probvalid & (delta[...,0]>1e-5)))
                if different.any():
                    pi,qi,kind,item=np.where(different)
                    flat=np.ravel_multi_index((np.full(len(pi),cell),np.full(len(pi),ti),pi,qi,kind,item),shape)
                    f=out/f'differences_d{d}_S{s}_{entry["treatment"]}.npz'
                    np.savez_compressed(f,flat_index=flat.astype(np.int32),donor_condition_id=a[pi],recipient_condition_id=b[pi],
                        content_id=np.array(meta['content_ids'])[qi],orbit_id=ids[qi],q=q[qi],query_type=kind.astype(np.int8),
                        query_item=item.astype(np.int8),patch_prediction=pa[different],donor_prediction=da[different],
                        donor_truth=truth[different],donor_native_wrong=(da!=truth)[different],
                        patch_probability=pp[different] if pp is not None else np.empty((0,7)),
                        donor_probability=dd[different] if dd is not None else np.empty((0,7)),
                        max_abs_probability_difference=delta[...,0][different],TV=delta[...,1][different],
                        is_answer_disagreement=(pa!=da)[different],is_probability_not_close=(probvalid & (delta[...,0]>1e-5))[different],
                        dataset=dataset,seed=seed,d=d,original_S=s,effective_suffix=entry['effective_suffix'],treatment=entry['treatment'])
                    detailed_count+=len(pi)
                print(json.dumps({'phase':'cache comparison','dataset':dataset,'seed':seed,'d':d,'S':s,'treatment':entry['treatment'],'paired':n,'missing':expected-n}),flush=True)
                del pp_all,dd_all,pp,dd
            flags.flush();diff.flush();answers.flush()
            np.savez_compressed(out/'orbit_blocks.npz',agree=block_agree,counts=block_n,orbit_ids=unique,
                content_id=np.array(meta['content_ids']),orbit_id=ids,directions=PAIRS,cell_d=np.repeat(DELAYS,3),cell_S=np.tile(SUFFIXES,2))
            del flags,diff,answers
    write_json(OUT/'input_index.json',{'entries':index_entries,'tensor_axes':['d/S cell','treatment','ordered nonself pair','canonical content','query_type','query_item'],
        'probability_difference_last_axis':['max_abs_difference','TV'],'answer_last_axis':['patch_prediction','donor_prediction','donor_truth','exact_equal_probability_elements'],
        'missing_answer_value':255,'flag_bits':FLAG,'semantic_decoding':'python -m experiments.memory_interchange.donor_consistency.inspect_record; keys are mapped through validated content/condition dictionaries, not positional assumptions'})
    csv_write(OUT/'pairing_coverage.csv',pairing);csv_write(OUT/'supplemental_controls.csv',control_rows)
    csv_write(OUT/'control_state_differences.csv',c_rows)
    write_json(OUT/'normalization_audit.json',norm_rows);write_json(OUT/'NN_audit.json',nn_rows)
    write_json(OUT/'compute_receipt.json',{'CPU_wall_seconds':time.perf_counter()-start,'GPU_seconds_added':0,
        'no_network_executed':True,'no_predictor_refitted':True,'missing_entries':missing,
        'explicit_full_probability_difference_records':detailed_count,
        'other_smaller_nonzero_probability_differences_preserved_in_tensors':True})

if __name__=='__main__':run()
