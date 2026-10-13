import hashlib
import json
import shutil
import numpy as np
import torch
from ..common import sha256,write_json
from ..v1_1.audit import evaluate,gate
from ..v1_1.data import load
from ..truth import contents
from .common import BASE,RESULTS,RUNS,Budget,config
from .train import initialize,optimizer_for,restore_checkpoint

def state_sha(value):
    digest=hashlib.sha256()
    def visit(obj):
        if torch.is_tensor(obj):
            v=obj.detach().cpu().contiguous();digest.update(str((str(v.dtype),tuple(v.shape))).encode());digest.update(v.numpy().tobytes())
        elif isinstance(obj,dict):
            for k in sorted(obj,key=repr):digest.update(repr(k).encode());visit(obj[k])
        elif isinstance(obj,(tuple,list)):
            digest.update(type(obj).__name__.encode())
            for v in obj:visit(v)
        else:digest.update(repr(obj).encode())
    visit(value);return digest.hexdigest()

def prepare():
    revision=config();out=RUNS/'pilot_seed0';out.mkdir(exist_ok=True)
    source=BASE/revision['pilot_checkpoint'];locked=out/'selected.pt'
    if not locked.exists():shutil.copy2(source,locked)
    assert sha256(source)==sha256(locked)
    if (RESULTS/'checkpoint_lock.json').exists():return
    with Budget():
        model=initialize(revision,'A');opt=optimizer_for(model,revision);rng=np.random.default_rng(100000)
        saved=restore_checkpoint(locked,model,opt,rng)
        assert saved['stage']==7 and saved['step']==49500 and saved['total_updates']==78000
        evaluations={}
        for split in ('T','V'):
            m,raw=evaluate(model,load('stage7_'+split))
            evaluations[split]=m;np.savez_compressed(RESULTS/f'best_fixed_{split}.npz',**raw)
    curve=json.loads((BASE/'runs/v1_2/candidate_A_seed0/stage_7/curve.json').read_text())
    coarse=dict(revision,validation={**revision['validation'],'gate_factors':revision['validation']['stage7_gate_factors']})
    previous=next(r for r in curve if r['step']==49000)
    best_record=next(r for r in curve if r['step']==49500)
    for split in ('T','V'):
        assert evaluations[split]['accuracy']==best_record[split+'_accuracy']
        assert abs(evaluations[split]['cross_entropy']-best_record[split+'_cross_entropy'])<=1e-6
    previous_pass=gate({'accuracy':previous['V_accuracy'],'strata':previous['V_strata']},coarse)
    current_pass=gate(evaluations['V'],coarse)
    manifest=json.loads((BASE/'split_manifest.json').read_text());q,ids=contents(manifest,'V')
    half=len(manifest['splits']['V']['orbit_ids'])//2
    geo_orbits=manifest['splits']['V']['orbit_ids'][:half]
    behavior_orbits=[];count=0
    for oid in manifest['splits']['V']['orbit_ids'][half:]:
        behavior_orbits.append(oid);count+=len(manifest['orbits'][oid]['members'])
        if count>=32:break
    geo_mask=np.isin(ids,geo_orbits);beh_mask=np.isin(ids,behavior_orbits)
    assert len(geo_orbits)==52 and len(behavior_orbits)==6 and not set(geo_orbits)&set(behavior_orbits)
    legacy=BASE/'runs/pilot_128/analysis'
    for filename,mask in [('geometry_states.npz',geo_mask),('behavior_states.npz',beh_mask)]:
        old=np.load(legacy/filename)
        assert np.array_equal(old['content'],q[mask]) and np.array_equal(old['orbit_id'],ids[mask])
    np.savez_compressed(RESULTS/'V_subset_manifest.npz',geometry_content=q[geo_mask],geometry_orbit_id=ids[geo_mask],
                        behavior_content=q[beh_mask],behavior_orbit_id=ids[beh_mask])
    result={'checkpoint':str(locked.resolve()),'source_checkpoint':str(source.resolve()),'sha256':sha256(source),
            'model_state_sha256':state_sha(saved['model']),'optimizer_state_sha256':state_sha(saved['optimizer']),
            'stage_step':saved['step'],'total_updates':saved['total_updates'],'fixed_evaluation':evaluations,
            'old_fine_gate_passed':gate(evaluations['V'],revision),'coarse_gate_passed':current_pass,
            'preceding_49000_coarse_gate_passed':previous_pass,'pilot_TASK_READY_v13':bool(current_pass and previous_pass),
            'no_seed0_training':True,'geometry_V_contents':int(geo_mask.sum()),'behavior_V_contents':int(beh_mask.sum()),
            'geometry_V_orbits':geo_orbits,'behavior_V_orbits':behavior_orbits,
            'legacy_subset_sources':{str(p.relative_to(BASE)):sha256(p) for p in (legacy/'geometry_states.npz',legacy/'behavior_states.npz',legacy/'mismatch.npz')},
            'prior_source_sha256':{str(p.relative_to(BASE)):sha256(p) for p in (BASE/'model.py',BASE/'v1_2/train.py',BASE/'v1_2/revision_v1_2.yaml')}}
    write_json(RESULTS/'checkpoint_lock.json',result)
    print(json.dumps({'locked_step':49500,'best_V':evaluations['V']['accuracy'],'pilot_TASK_READY_v13':result['pilot_TASK_READY_v13']}),flush=True)

if __name__=='__main__':prepare()
