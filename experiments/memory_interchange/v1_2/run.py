import json
import subprocess
import sys
import time
import numpy as np
import torch
from ..common import sha256, write_json
from ..train import load_model
from ..v1_1.audit import evaluate, gate
from ..v1_1.data import load
from .common import BASE, ROOT, RESULTS, RUNS, Budget, config
from .prepare import prepare
from .train import curriculum
from .ready import measure as ready_measure
from .science import scientific_config, measure as scientific_measure, predictions
from .extension import measure as extension_measure

def preservation():
    record=json.loads((RESULTS/'preserved_versions.json').read_text())
    bad=[name for name,v in record['files'].items() if sha256(BASE/name)!=v['sha256']]
    assert not bad, f'Preserved versions changed: {bad}'
    assert sha256(BASE/'split_manifest.json')==record['split_sha256']
    write_json(RESULTS/'preservation_check.json',{'all_unchanged':True,'files_checked':len(record['files']),
              'split_sha256':record['split_sha256'],'formal_B_contents':134})

def freeze():
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and (p.suffix in ('.py','.yaml','.txt') or p.name in ('CHANGES.md','README.md'))]
    old=json.loads((RESULTS/'preserved_versions.json').read_text())
    snapshot={'version':'1.2','freeze_type':'before_formal_G_B','selected_candidate':'A',
              'initialization':'pytorch_default','source_sha256':{str(p.relative_to(BASE)):sha256(p) for p in files},
              'original_scientific_source_sha256':{n:v['sha256'] for n,v in old['files'].items()
                                                    if n.endswith('.py') or n=='config.yaml'},
              'split_sha256':sha256(BASE/'split_manifest.json'),
              'datasets_manifest_sha256':sha256(BASE/'results/v1_1/datasets/manifest.json'),
              'formal_seeds':config()['formal_seeds'],'maximum_updates_per_seed':200000}
    path=RESULTS/'frozen_revision.json'
    if path.exists():
        previous=json.loads(path.read_text())
        assert previous==snapshot, 'Frozen sources changed; no result-driven revision permitted'
    else:write_json(path,snapshot)

def selected_validation(run, revision):
    result=json.loads((run/'summary.json').read_text())
    if 'selected_checkpoint' not in result:
        sid=result['failed_stage']
        path=run/f'stage_{sid}'/'best.pt'
        result['selected_checkpoint']=str(path.relative_to(BASE))
        result['selection_note']='Curriculum failed; last attempted stage V-best retained, no replacement seed'
        write_json(run/'summary.json',result)
    model,saved=load_model(BASE/result['selected_checkpoint'])
    with Budget():
        val,raw=evaluate(model,load('original_V'),revision['execution']['evaluation_batch'])
        np.savez_compressed(run/'selected_original_V_predictions.npz',**raw)
    check={'selected_checkpoint':result['selected_checkpoint'],'selected_stage':saved['stage'],
           'selected_step':saved['step'],'original_V':val,'original_V_factor_gate_passed':gate(val,revision),
           'curriculum_passed':result['passed']}
    write_json(run/'selected_validation.json',check)
    return check

def execute():
    revision=config();preservation();prepare()
    tested=RESULTS/'tests.txt'
    if not tested.exists():
        with Budget():
            p=subprocess.run([sys.executable,'-m','pytest',str(BASE/'tests'),str(BASE/'v1_1/tests'),str(ROOT/'tests'),'-q'],capture_output=True,text=True)
        tested.write_text(p.stdout+p.stderr)
        if p.returncode:
            write_json(RESULTS/'status.json',{'status':'BLOCKED','reason':'implementation_tests_failed',
                       'formal_G_B_opened':False,'H1':'not_tested','H2':'not_tested','H3':'not_tested'})
            raise RuntimeError(p.stdout+p.stderr)
    pilot=curriculum('A',revision)
    if not pilot['passed']:
        write_json(RESULTS/'status.json',{'status':'BLOCKED','reason':'pilot_stage_budget_exhausted',
                    'failed_stage':pilot['failed_stage'],'stage_cap':50000,'pilot_total_updates':pilot['total_updates'],
                    'formal_seeds_executed':[],'formal_G_B_opened':False,
                    'H1':'not_tested','H2':'not_tested','H3':'not_tested'})
        return
    ready=ready_measure(pilot,revision)
    write_json(RESULTS/'status.json',{'status':'TASK_READY','pilot_original_V_accuracy':ready['original_V']['accuracy'],
               'formal_seeds_executed':[],'formal_G_B_opened':False,'H1':'not_tested','H2':'not_tested','H3':'not_tested'})
    freeze()
    estimate=json.loads((RESULTS/'training_cost_estimate.json').read_text())
    # Only after TASK_READY: allow generous evaluation/cache/save/analysis margin.
    # This is an estimate, not a replacement for persistent actual budget checks.
    train_seconds=estimate['formal_training_seconds_estimate']
    cost={'formal_max_updates':1200000,'measured_training_seconds':train_seconds,
          'validation_and_save_multiplier':1.5,'science_margin_seconds_per_seed':600,
          'conservative_estimated_seconds':1.5*train_seconds+6*600,
          'remaining_budget_seconds':Budget().remaining(),
          'scientific_labels_not_computed_yet':True,'actual_cost_accounted_separately':True}
    write_json(RESULTS/'formal_cost_estimate.json',cost)
    if cost['conservative_estimated_seconds']>Budget().remaining():
        write_json(RESULTS/'status.json',{'status':'BLOCKED','reason':'remaining_12_hour_budget_insufficient',
                   'formal_G_B_opened':False,'formal_seeds_executed':[],
                   'H1':'not_tested','H2':'not_tested','H3':'not_tested'})
        return
    formal=[]
    # Complete all task training and choose all checkpoints before G/B is opened.
    for seed in revision['formal_seeds']:
        result=curriculum('A',revision,seed=seed,pilot=False)
        check=selected_validation(RUNS/f'seed_{seed}',revision)
        formal.append({'seed':seed,'curriculum_passed':result['passed'],
                       'total_updates':result['total_updates'],'original_V_factor_gate_passed':check['original_V_factor_gate_passed']})
        write_json(RESULTS/'formal_training_summary.json',formal)
        print(json.dumps({'formal_training_complete':seed,**formal[-1]}),flush=True)
    preservation();freeze()
    cfg=scientific_config();manifest=json.loads((BASE/'split_manifest.json').read_text())
    for seed in cfg['seeds']:
        scientific_measure(RUNS/f'seed_{seed}',cfg,manifest)
    prediction=predictions(cfg)
    valid=all(s['curriculum_passed'] and s['original_V_factor_gate_passed'] for s in formal)
    valid &= all(json.loads((RUNS/f'seed_{seed}'/'analysis/intervention_summary.json').read_text())['native_strata_passed'] for seed in cfg['seeds'])
    if valid:
        for seed in cfg['seeds']:
            run=RUNS/f'seed_{seed}'
            if not (run/'extension/summary.csv').exists():extension_measure(run,cfg)
    write_json(RESULTS/'status.json',{'status':'COMPLETED' if valid else 'EXPLORATORY',
               'reason':'all_prespecified_analyses_complete' if valid else 'formal_behavior_gates_insufficient_all_seeds_retained',
               'formal_seeds_executed':cfg['seeds'],'formal_G_B_opened':True,
               'formal_behavior_gates_all_passed':bool(valid),'extension_executed':bool(valid),
               'H1':'see_formal_interventions','H2':'see_held_out_prediction','H3':'see_held_out_prediction'})
    preservation()

def main():
    execute()
    from .final_audit import augment
    augment()
    from .report import build_report
    build_report()

if __name__=='__main__':main()
