"""Inspect metadata, verify actual keys/input schedules, then freeze before scoring."""
import time
import numpy as np
from ..truth import CONDITIONS, contents, preimage, matrix_apply
from ..task import prefix_inputs, suffix_inputs, trials
from ..common import config
from .common import *

def freeze():
    if (OUT/'analysis_plan.json').exists():load_plan();return
    previous=read_json(FINAL/'preservation_manifest.json')
    previous.update(read_json(FINAL/'artifact_manifest.json'))
    for p in [FINAL/'artifact_manifest.json',FINAL/'preservation_manifest.json',BASE/'split_manifest.json']:
        previous[str(p.relative_to(BASE))]={'sha256':sha(p),'bytes':p.stat().st_size}
    write_json(OUT/'preservation_manifest.json',previous)
    preservation=preserve_check();assert preservation['all_unchanged'],preservation
    manifest=read_json(BASE/'split_manifest.json');cfg=config()
    allq,allids=contents(manifest,'B');b1,b1ids=contents(manifest,'B',True)
    keep=~np.isin(allids,manifest['behavior_orbit_ids'])
    pools={'B1':(b1,b1ids),'B_reserved':(allq[keep],allids[keep])}
    saved=np.load(FINAL/'B_reserved_members.npz')
    assert np.array_equal(pools['B_reserved'][0],saved['content'])
    assert np.array_equal(pools['B_reserved'][1],saved['orbit_id'])
    assert len(b1)==134 and len(set(b1ids))==19
    assert len(pools['B_reserved'][0])==362 and len(set(pools['B_reserved'][1]))==51
    assert set(map(tuple,b1)).isdisjoint(set(map(tuple,pools['B_reserved'][0])))
    inputs={};entries=[];state_metadata=[];datasets={};missing=[]
    def add(path):
        key=str(Path(path).relative_to(BASE))
        if key not in inputs:inputs[key]=record(path)
        return inputs[key]
    for p in [BASE/'split_manifest.json',BASE/'config.yaml',FINAL/'frozen_plan.json',FINAL/'README.md',FINAL/'report.md',FINAL/'aggregation_note.json',FINAL/'B_reserved_members.npz']:
        add(p)
    for dataset,(q,ids) in pools.items():
        datasets[dataset]={'contents':len(q),'orbits':len(set(ids)),'members':q.tolist(),
            'content_ids':member_ids(q).tolist(),'orbit_ids_per_content':ids.tolist(),
            'orbit_ids':sorted(map(int,set(ids))),'already_opened':True,'analysis_role':'post hoc descriptive; not independent confirmation'}
        for seed in SEEDS:
            sp=state_path(dataset,seed);st=add(sp)
            if not st['exists']:
                missing.append({'dataset':dataset,'seed':seed,'reason':'natural state metadata missing','path':st['path']})
                state_metadata.append({'dataset':dataset,'seed':seed,'valid':False});continue
            f=np.load(sp);cq=f['content'];cids=member_ids(cq)
            ck=[condition_key(r,h) for r,h in zip(f['contexts'],f['histories'])]
            expected=[condition_key(r,h) for r,h in CONDITIONS]
            assert len(ck)==18 and len(set(ck))==18 and ck==expected
            assert len(cids)==len(set(map(int,cids)))
            canonical=member_ids(q);lookup={int(key):i for i,key in enumerate(cids)}
            expected_orbit={int(key):int(orb) for key,orb in zip(canonical,ids)}
            assert np.array_equal(f['orbit_id'],[expected_orbit[int(key)] for key in cids])
            qindex=[lookup.get(int(key),-1) for key in canonical]
            assert set(map(int,cids)).issubset(set(map(int,canonical)))
            assert np.array_equal(f['delays'],DELAYS)
            assert np.array_equal(f['cache_steps'],[21,31])
            for di,d in enumerate(DELAYS):
                for a,(r,h) in enumerate(CONDITIONS):
                    assert np.array_equal(f['initial'][di,a],preimage(cq,h,r))
                    assert np.array_equal(matrix_apply(f['initial'][di,a],h,r),cq)
            state_metadata.append({'dataset':dataset,'seed':seed,'valid':True,'path':st['path'],
                'condition_keys':ck,'canonical_to_cache_q_index':qindex,
                'content_ids':cids.tolist(),'orbit_id':f['orbit_id'].tolist(),
                'missing_content_ids':[int(key) for key,idx in zip(canonical,qindex) if idx<0]})
            for d in DELAYS:
                for s in SUFFIXES:
                    native=add(native_path(dataset,seed,d,s));patch=add(patch_path(dataset,seed,d,s))
                    pschema=patch['fields'];nschema=native['fields']
                    if patch['exists']:
                        pf=np.load(BASE/patch['path']);pq=pf['content'];pids=member_ids(pq)
                        assert len(pids)==len(set(map(int,pids)))
                        assert set(map(int,pids)).issubset(set(map(int,canonical)))
                        assert np.array_equal(pf['orbit_id'],[expected_orbit[int(key)] for key in pids])
                        assert int(pf['seed'])==seed and int(pf['d'])==d
                        assert str(pf['S'])==s and str(pf['translated_S'])==SWAP[s]
                        plookup={int(key):i for i,key in enumerate(pids)}
                        pindex=[plookup.get(int(key),-1) for key in canonical]
                    else:pindex=[-1]*len(q)
                    for ti,treatment in enumerate(TREATMENTS):
                        e=s if ti==0 else SWAP[s]
                        enative=add(native_path(dataset,seed,d,e));eh=enative['fields']
                        pk=f'patch_{treatment}_logp';ak=f'patch_{treatment}_mapping_predictions'
                        nf='native_logp' if 'native_logp' in eh else 'natural_logp' if 'natural_logp' in eh else None
                        pf=pk if pk in pschema else ak if ak in pschema else None
                        for field,shape in [(pf,[18,18,len(pq) if patch['exists'] else len(q),2,7,7] if pf==pk else [18,18,len(pq) if patch['exists'] else len(q),2,7]),
                                            (nf,[18,len(cq),2,7,7])]:
                            schema=pschema if field==pf else eh
                            if field:assert schema[field]['shape']==shape,(patch['path'],field,schema[field],shape)
                        entry={'dataset':dataset,'seed':seed,'d':d,'original_S':s,'treatment':treatment,'effective_suffix':e,
                            'patch_path':patch['path'],'patch_field':pf,'patch_probability':pf==pk,
                            'native_path':enative['path'],'native_field':nf,'native_probability':nf is not None,
                            'patch_q_index':pindex,'native_q_index':qindex,
                            'input_signature_key':e,'available':pf is not None and nf is not None,
                            'expected_queries_nonself':306*len(q)*14,'expected_diagonal_control_queries':18*len(q)*14,
                            'condition_axis_mapping':'validated natural state metadata + preserved writer source',
                            'classes':list(range(7))}
                        entries.append(entry)
                        if not entry['available']:missing.append({k:entry[k] for k in ('dataset','seed','d','original_S','treatment','patch_path','native_path','patch_field','native_field')})
    # Construct input arrays only. No network, no output probabilities are read.
    schedules=[];kind=np.repeat([0,1],7);item=np.tile(np.arange(7),2)
    for e in SUFFIXES:
        future=suffix_inputs([e]*14,kind,item,cfg)
        assert np.all(future['ctrl'][...,:2]==0) and np.all(future['stim']==0)
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in future.items()}
        for d in DELAYS:
            for r,h in CONDITIONS:
                p=np.repeat(preimage(b1[:1],h,r),14,axis=0)
                full=trials(p,np.full(14,r),np.array([h+e]*14),np.full(14,d),kind,item,cfg)
                start=21+d
                for key in ('stim','ctrl','phase'):assert np.array_equal(full[key][:,start:start+16],future[key])
                assert np.array_equal(full['query'],future['query']) and np.all(full['lengths']==37+d)
        schedules.append({'effective_suffix':e,'input_sha256':hashes,'post_swap_steps':16,
            'cue_relative_steps_1based':[1,5],'rule_blank_steps':3,'final_hold_steps':5,
            'query_relative_steps_1based':[14,15,16],'external_context_after_swap':False,
            'same_input_checked_for_all_conditions_delays':True})
    checkpoints=[]
    oldplan=read_json(FINAL/'frozen_plan.json')
    for c in oldplan['checkpoints']:
        rec=add(BASE/c['path']);assert rec['sha256']==c['sha256'];checkpoints.append(c)
    for seed in SEEDS:add(RUNS/f'seed_{seed}/analysis/baseline_controls.json')
    # The independent sample is chosen from keys BEFORE output readout.
    sample=[]
    for dataset,(q,ids) in pools.items():
        for seed in SEEDS:
            for cross in (False,True):
                for same_history in (False,True):
                    eligible=[(a,b) for a,b in PAIRS if (CONDITIONS[a][0]!=CONDITIONS[b][0])==cross and
                              (CONDITIONS[a][1]==CONDITIONS[b][1])==same_history]
                    if not eligible:continue
                    a,b=map(int,eligible[0]);qi=(seed-101)%len(q)
                    for d in DELAYS:
                        for s in SUFFIXES:
                            for ti,treatment in enumerate(TREATMENTS):
                                for k in (0,1):
                                    sample.append({'dataset':dataset,'seed':seed,'donor':a,'recipient':b,
                                        'content_id':int(member_ids(q[qi:qi+1])[0]),'q':q[qi].tolist(),'d':d,'S':s,
                                        'treatment':treatment,'effective_suffix':s if ti==0 else SWAP[s],
                                        'query_type':k,'query_item':(seed-101+d+k)%7})
    code={str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.glob('*.py'))}
    for p in [BASE/'truth.py',BASE/'task.py',BASE/'intervene.py',BASE/'cache.py',BASE/'v1_3/controls.py',BASE/'final_audit/behavior.py']:add(p)
    plan={'date':'2026-10-13','version':'donor consistency 1.0','role':'new descriptive audit on already opened B1/B_reserved',
        'freeze_scope':'This analysis only; frozen before computing its consistency metrics, not retrospective preregistration of prior findings.',
        'AGENTS_read':'No AGENTS.md found in workspace or applicable ancestors after explicit search',
        'project_base':str(BASE),'readmes':['README.md','v1_3/README.md','Final_audit/README.md'],
        'datasets':datasets,'conditions':[{'id':i,'r':r,'H':h,'key':condition_key(r,h)} for i,(r,h) in enumerate(CONDITIONS)],
        'native_pairing_key':['dataset','seed','donor_condition(r,H)','content_id(permutation rank)','d','swap_event_step','effective_suffix','query_type','query_item'],
        'patch_key_additions':['recipient_condition(r,H)','treatment','original_S'],
        'controlled_expansion':'exactly one unique native key per patch; with complete coverage each native key expands to 17 recipients × 2 treatments, plus 2 diagonal controls',
        'main_directions':PAIRS.tolist(),'diagonals':'all 18 saved, excluded from primary inter-condition mean and explicitly retained as controls',
        'argmax':'lowest index among tied maxima, class order 0..6; same for both sides',
        'class_order':list(range(7)),'query_types':{'0':'item→position','1':'position→item'},
        'probabilities':'float64 exp(saved logp), check finite and raw sum within 2e-6, divide by raw sum; compare normalized P on last class axis',
        'probability_tolerances':[1e-6,1e-5,1e-4],'primary_close_tolerance':1e-5,
        'raw_normalization_atol':2e-6,'normalized_atol':3e-15,
        'independent_probability_reduction_atol':3e-15,
        'exact_equality':'finite normalized probability elements compared exactly; per-element and all-7 query rates separate',
        'aggregation':'primary all: equal six seeds, equal ordered directions, d/S/query types; within cells all content-query observations pooled. Record numerator/denominator separately. Donor correct/error subsets keep eligible query counts and additionally report equal nonempty-cell rates; descriptive distribution quantiles are pooled queries, explicitly distinct from equal-seed/cell means.',
        'strata':'all fixed factors including empty same-context/same-history nonself grid; no results-driven selection; S is original S, effective_suffix saved',
        'bootstrap':{'rng_seed':20261013,'replicates':10000,'datasets':'independent generator reset to same seed per dataset',
            'unit':'19 B1 or 51 B_reserved whole orbits, common draws for ALL 6 fixed models/directions/treatments/d/S/queries',
            'seed_resampling':False,'conditioning':'six frozen models; no retraining or unknown-model population uncertainty'},
        'full_vector':'all 14 argmax outputs equal; item-7 and position-7 separately; retain non-bijective predictions, do not conflate with permutation correctness',
        'donor_truth':'F_a(effective_suffix)(q), used only to stratify actual native output correctness, never to fill missing predictions',
        'missing_policy':'use all valid existing pairable records; absent prediction gives missing count; absent P but actual answers present gives answer-only; no network replay',
        'compact_records':'all query flags/differences/predictions + full semantic input index; explicit 7-class P pairs for answer disagreements or max P difference>1e-5; all smaller nonzero differences remain indexed and retrievable without duplicating original P arrays',
        'input_files':inputs,'state_metadata':state_metadata,'entries':entries,'input_schedules':schedules,
        'checkpoints':checkpoints,'fixed_direct_index_sample':sample,'code_sha256':code,
        'no_network_execution':True,'no_probe_or_predictor_refit':True,'missing_conditions_at_freeze':missing}
    write_json(OUT/'analysis_plan.json',plan)
    write_json(OUT/'freeze_receipt.json',{'unix_time':time.time(),'analysis_plan_sha256':sha(OUT/'analysis_plan.json'),
        'no_consistency_metrics_read_yet':True,'preservation':preservation})
    print(json.dumps({'plan_frozen':True,'datasets':{k:[v['contents'],v['orbits']] for k,v in datasets.items()},'entries':len(entries),'missing':len(missing),'old_files_verified':preservation['files']}),flush=True)

if __name__=='__main__':freeze()
