import json
import pickle
import random
import shutil
import time
import numpy as np
import torch
from ..common import rng_state, restore_rng, sha256, write_json
from ..v1_1.audit import evaluate
from ..v1_1.data import load
from .common import BASE, RESULTS, RUNS, Budget, config
from .train import initialize, optimizer_for, restore_checkpoint, train_step
from ..v1_1.common import model_config
from ..v1_1.data import generate
from ..truth import contents
from .diagnostics import diagnose

def same_rng(a,b):
    return (a['python']==b['python'] and a['generator']==b['generator'] and
            a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and
            a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['torch'].cpu(),b['torch'].cpu()) and
            len(a['cuda'])==len(b['cuda']) and all(torch.equal(x.cpu(),y.cpu()) for x,y in zip(a['cuda'],b['cuda'])))

def prepare():
    revision=config(); out=RUNS/'candidate_A_seed0';out.mkdir(parents=True,exist_ok=True)
    source=BASE/revision['resume']['checkpoint']
    old=source.parent.parent
    for name in ['initial.pt']:
        if not (out/name).exists():shutil.copy2(old/name,out/name)
    for j in range(1,6):
        if not (out/f'stage_{j}').exists():shutil.copytree(old/f'stage_{j}',out/f'stage_{j}')
    stage_out=out/'stage_6';stage_out.mkdir(exist_ok=True)
    for name in ('latest.pt','best.pt','curve.json'):
        if not (stage_out/name).exists():shutil.copy2(old/'stage_6'/name,stage_out/name)
    if (RESULTS/'restore_consistency.json').exists():return
    model=initialize(revision,'A'); opt=optimizer_for(model,revision);rng=np.random.default_rng(100000)
    saved=restore_checkpoint(source,model,opt,rng)
    assert saved['step']==5000 and saved['stage']==6 and saved['total_updates']==18500
    assert all(int(v['step'])==18500 for v in opt.state.values())
    assert same_rng(saved['rng'],rng_state(rng))
    for key,value in model.state_dict().items():assert torch.equal(value.cpu(),saved['model'][key].cpu())
    curve=json.loads((source.parent/'curve.json').read_text());assert curve[-1]['step']==saved['step']
    old_summary=json.loads((source.parent/'summary.json').read_text())
    results={}
    with Budget():
        for split in ('T','V'):
            actual,raw=evaluate(model,load('stage6_'+split),revision['execution']['evaluation_batch'])
            expected=old_summary['last_validation' if split=='V' else 'last_training_evaluation']
            differences={'accuracy':abs(actual['accuracy']-expected['accuracy']),
                         'cross_entropy':abs(actual['cross_entropy']-expected['cross_entropy'])}
            for factor,cells in actual['strata'].items():
                for label,cell in cells.items():
                    ref=expected['strata'][factor][label];assert cell['n']==ref['n']
                    assert abs(cell['accuracy']-ref['accuracy'])<=revision['resume']['accuracy_absolute_tolerance']
                    assert abs(cell['cross_entropy']-ref['cross_entropy'])<=revision['resume']['CE_absolute_tolerance']
            assert differences['accuracy']<=revision['resume']['accuracy_absolute_tolerance']
            assert differences['cross_entropy']<=revision['resume']['CE_absolute_tolerance']
            results[split]={'actual':actual,'absolute_difference':differences,'passed':True}
            np.savez_compressed(RESULTS/f'restored_stage6_{split}.npz',**raw)
        diagnose(model,revision,6,RESULTS/'two_step_diagnostics','restore_step5000')
        # Measure unchanged stochastic training on an in-memory clone, then restore
        # model, Adam and every RNG. These benchmark steps are never promoted.
        restore_checkpoint(source,model,opt,rng)
        pool=contents(json.loads((BASE/'split_manifest.json').read_text()),'T')[0]
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(200):train_step(model,opt,generate(pool,128,rng,revision,revision['stages'][5]),revision)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        restore_checkpoint(source,model,opt,rng)
        assert same_rng(saved['rng'],rng_state(rng))
    result={'passed':True,'source_checkpoint':str(source.resolve()),'source_sha256':sha256(source),
            'stage_step':saved['step'],'total_updates':saved['total_updates'],
            'optimizer_all_parameter_step':18500,'weights_exact':True,'all_rng_exact':True,
            'checkpoint_kind':'latest','old_report_kind':'stage6 endpoint/latest; best also step5000',
            'numerical_tolerances':revision['resume'],'fixed_evaluation':results}
    write_json(RESULTS/'restore_consistency.json',result)
    throughput=200/seconds
    write_json(RESULTS/'training_cost_estimate.json',{'benchmark_updates':200,'benchmark_seconds':seconds,
                'updates_per_second':throughput,'benchmark_changes_discarded_all_RNG_restored':True,
                'pilot_max_additional_updates':95000,'pilot_training_seconds_estimate':95000/throughput,
                'formal_max_updates':1200000,'formal_training_seconds_estimate':1200000/throughput,
                'validation_save_cache_analysis_not_in_benchmark':True,
                'inherited_gpu_seconds':Budget().inherited,'total_limit_seconds':Budget().limit})
    write_json(RESULTS/'previous_stage6_curve.json',curve)
    recent=[r for r in curve if r['step']>=3000]
    write_json(RESULTS/'previous_curve_trend.json',{'whole_stage_points':len(curve),'recent_2000':recent,
                'V_accuracy_change':recent[-1]['V_accuracy']-recent[0]['V_accuracy'],
                'V_CE_change':recent[-1]['V_cross_entropy']-recent[0]['V_cross_entropy'],
                'k2_accuracy_change':recent[-1]['V_strata']['non_N']['2']['accuracy']-recent[0]['V_strata']['non_N']['2']['accuracy'],
                'description':'All ten sampled V accuracy/CE points improve; no plateau assertion or causal interpretation.'})
    print(json.dumps({'restore_passed':True,'source_sha256':sha256(source),'measured_updates_per_second':throughput}),flush=True)

if __name__=='__main__':prepare()
