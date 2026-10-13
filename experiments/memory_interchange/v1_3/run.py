import json
import subprocess
import sys
import time
import numpy as np
from ..common import sha256,write_json
from ..train import load_model
from ..v1_1.audit import evaluate,gate
from ..v1_1.data import load
from .common import BASE,ROOT,RESULTS,RUNS,Budget,config
from .prepare import prepare
from .science import exploratory,measure
from .train import curriculum

def preservation():
    r=json.loads((RESULTS/'preserved_versions.json').read_text())
    bad=[n for n,v in r['files'].items() if sha256(BASE/n)!=v['sha256']]
    assert not bad,bad
    write_json(RESULTS/'preservation_check.json',{'all_unchanged':True,'files_checked':len(r['files']),'split_sha256':sha256(BASE/'split_manifest.json')})

def freeze():
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='report.md']
    snapshot={'version':'1.3','source_sha256':{str(p.relative_to(BASE)):sha256(p) for p in files},
              'split_sha256':sha256(BASE/'split_manifest.json'),'old_versions_manifest_sha256':sha256(RESULTS/'preserved_versions.json'),
              'datasets_manifest_sha256':sha256(BASE/'results/v1_1/datasets/manifest.json'),
              'pilot_checkpoint_sha256':sha256(RUNS/'pilot_seed0/selected.pt'),
              'V_not_a_confirmatory_dataset':True,'formal_seeds':list(range(101,107)),
              'primary_estimand':'L_all; M_CKA_v13 versus M0_old+D_input on LOSO unseen_cross',
              'masks_from_natural_outputs_before_patch':True}
    path=RESULTS/'frozen_plan.json'
    if path.exists():assert json.loads(path.read_text())==snapshot,'Frozen source changed'
    else:write_json(path,snapshot)

def selected_validation(run):
    r=json.loads((run/'summary.json').read_text())
    if 'selected_checkpoint' not in r:
        r['selected_checkpoint']=str((run/f"stage_{r['failed_stage']}"/'best.pt').relative_to(BASE))
        r['selection_note']='Failed curriculum retained; V-best last attempted stage, no replacement seed'
        write_json(run/'summary.json',r)
    path=run/'selected_validation.json'
    if path.exists():return json.loads(path.read_text())
    cfg=config();coarse=dict(cfg,validation={**cfg['validation'],'gate_factors':cfg['validation']['stage7_gate_factors']})
    with Budget():
        model,saved=load_model(BASE/r['selected_checkpoint']);v,raw=evaluate(model,load('original_V'))
        np.savez_compressed(run/'selected_original_V.npz',**raw)
    result={'original_V':v,'coarse_gate_passed':gate(v,coarse),'old_fine_gate_passed':gate(v,cfg),
            'checkpoint':r['selected_checkpoint'],'checkpoint_sha256':sha256(BASE/r['selected_checkpoint']),
            'selected_stage':saved['stage'],'selected_stage_step':saved['step'],'curriculum_passed':r['passed']}
    write_json(path,result);return result

def execute():
    preservation();prepare();cfg=config()
    # Initial 23-test execution was run before this driver, and is charged once.
    charges=RESULTS/'initial_test_cost.json'
    if not charges.exists():
        used=Budget().used;extra=6.91
        write_json(RESULTS/'budget.json',{'gpu_seconds':used+extra,'inherited_gpu_seconds':Budget().inherited,
                   'v1_3_gpu_seconds':used+extra-Budget().inherited,'limit_seconds':43200})
        write_json(charges,{'seconds':extra,'reason':'conservative complete wall time of initial 23-test execution'})
    if not (RESULTS/'final_tests.txt').exists():
        with Budget():
            p=subprocess.run([sys.executable,'-m','pytest',str(BASE/'tests'),str(BASE/'v1_1/tests'),str(BASE/'v1_2/tests'),str(ROOT/'tests'),'-q'],capture_output=True,text=True)
        (RESULTS/'final_tests.txt').write_text(p.stdout+p.stderr)
        assert not p.returncode,p.stdout+p.stderr
    exploratory()
    from .report import exploratory as exploratory_report
    exploratory_report()
    lock=json.loads((RESULTS/'checkpoint_lock.json').read_text())
    if not lock['pilot_TASK_READY_v13']:
        write_json(RESULTS/'status.json',{'status':'BLOCKED','reason':'locked_pilot_coarse_factors_failed','formal_G_B_opened':False})
        return
    freeze()
    original_speed=json.loads((BASE/'results/v1_2/training_cost_estimate.json').read_text())['updates_per_second']
    estimate=1.5*1200000/original_speed+3600
    write_json(RESULTS/'formal_cost_estimate.json',{'max_updates':1200000,'conservative_estimated_seconds':estimate,
               'remaining_budget_seconds':Budget().remaining(),'V_diagnostics_completed':True,'G_B_not_opened':True})
    if Budget().remaining()<estimate:
        write_json(RESULTS/'status.json',{'status':'BLOCKED','reason':'insufficient_remaining_12_GPU_hours','formal_G_B_opened':False});return
    formal=[]
    # No formal model sees G/B or exchange outcomes while learning.
    for seed in cfg['formal_seeds']:
        r=curriculum('A',cfg,seed=seed,pilot=False);check=selected_validation(RUNS/f'seed_{seed}')
        formal.append({'seed':seed,'curriculum_passed':r['passed'],'coarse_gate_passed':check['coarse_gate_passed'],
                       'old_fine_gate_passed':check['old_fine_gate_passed'],'updates':r['total_updates']})
        write_json(RESULTS/'formal_training_summary.json',formal)
        write_json(RESULTS/'progress.json',{'phase':'formal_training','finished_seeds':[x['seed'] for x in formal],'formal_G_B_opened':False})
        print(json.dumps({'formal_seed_complete':seed,**formal[-1]}),flush=True)
    preservation();freeze()
    for seed in cfg['formal_seeds']:
        run=RUNS/f'seed_{seed}';r=json.loads((run/'summary.json').read_text())
        measure(BASE/r['selected_checkpoint'],run/'initial.pt',run/'analysis',seed,False)
        write_json(RESULTS/'progress.json',{'phase':'formal_science','latest_seed':seed,'formal_G_B_opened':True})
    from .predict import run as predict
    predict()
    # The original four-step extension remains conditional on native main quality.
    extension_ok=all(x['curriculum_passed'] and x['coarse_gate_passed'] for x in formal)
    extension_ok &= all(json.loads((RUNS/f'seed_{s}/analysis/intervention_summary.json').read_text())['native_strata_passed'] for s in cfg['formal_seeds'])
    if extension_ok:
        from .extension import measure as extend
        from ..common import config as original_config
        for seed in cfg['formal_seeds']:
            run=RUNS/f'seed_{seed}'
            if not (run/'extension/summary.csv').exists():extend(run,original_config())
    write_json(RESULTS/'status.json',{'status':'COMPLETED','reason':'all_v13_prespecified_formal_computations_complete',
               'formal_G_B_opened':True,'formal_seeds':cfg['formal_seeds'],'formal_new_gate_all_passed':all(x['coarse_gate_passed'] for x in formal),
               'old_fine_gate_all_passed':all(x['old_fine_gate_passed'] for x in formal),'failed_seeds_retained':True,
               'extension_executed':bool(extension_ok),'science_support_is_not_a_completion_condition':True})
    preservation();freeze()

def main():
    execute()
    from .report import final
    final()

if __name__=='__main__':main()
