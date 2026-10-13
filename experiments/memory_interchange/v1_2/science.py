"""Unchanged v1.0 scientific estimands, with v1.2 paths and accounting."""
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from ..cache import cache_states
from ..common import config as original_config, seed_all, write_json
from ..geometry import fit_probe, probe_scores, linear_cka, raw_distance, procrustes_distance
from ..intervene import evaluate_interventions
from ..train import load_model
from ..truth import contents
from ..predict import (merged_rows, leave_seed_out, hierarchical_bootstrap, stratified_spearman,
                       quantile_examples, save_csv)
from .common import BASE, RESULTS, RUNS, Budget

def scientific_config():
    return original_config()

def measure(run, cfg, manifest):
    run=Path(run);out=run/'analysis';out.mkdir(exist_ok=True)
    choice=json.loads((run/'summary.json').read_text())
    checkpoint=BASE/choice['selected_checkpoint']
    assert choice['seed'] in cfg['seeds']
    seed_all(123,cfg['execution']['cpu_threads'],cfg['execution']['deterministic'])
    model,saved=load_model(checkpoint);assert saved['seed']==choice['seed']
    if not (out/'intervention_summary.json').exists():
        torch.cuda.reset_peak_memory_stats()
        q,ids=contents(manifest,'B',behavior=True);assert len(q)==134
        with Budget() as budget:
            budget.check(60)
            cached=cache_states(model,q,ids,cfg,out/'behavior_states.npz')
            evaluate_interventions(model,cached,cfg,out,saved['seed'])
    if not (out/'geometry_summary.json').exists():
        started=time.perf_counter();q,ids=contents(manifest,'G')
        with Budget() as budget:
            budget.check(60)
            cached=cache_states(model,q,ids,cfg,out/'geometry_states.npz')
            probe=fit_probe(model,manifest,cfg,out/'probe.npz')
            initial,_=load_model(run/'initial.pt')
            init=cache_states(initial,q,ids,cfg,out/'initial_geometry_states.npz')
        item,whole=probe_scores(cached,probe,cfg['n']);rows=[]
        for di,d in enumerate(cfg['timing']['delays']):
            for a,b in itertools.combinations(range(18),2):
                x,y=cached['memory'][di,a],cached['memory'][di,b]
                rows.append({'seed':saved['seed'],'a':a,'b':b,'d':d,'cka':linear_cka(x,y),
                             'raw':raw_distance(x,y),'proc':procrustes_distance(x,y),
                             'init_cka':linear_cka(init['memory'][di,a],init['memory'][di,b]),
                             'control_distance':raw_distance(cached['control'][di,a:a+1],cached['control'][di,b:b+1]),
                             'probe_item_acc':float((item[di,a]+item[di,b])/2),
                             'probe_perm_acc':float((whole[di,a]+whole[di,b])/2)})
        save_csv(out/'geometry.csv',rows)
        write_json(out/'geometry_summary.json',{'seed':saved['seed'],'split':'G','contents':len(q),
                   'geometry_pairs':len(rows),'probe_item_accuracy':item.tolist(),
                   'probe_permutation_accuracy':whole.tolist(),'elapsed_seconds':time.perf_counter()-started,
                   'undefined_values':sum(not np.isfinite(row[k]) for row in rows for k in ('cka','proc','init_cka'))})

def predictions(cfg):
    runs=[RUNS/f'seed_{seed}' for seed in cfg['seeds']]
    out=RESULTS/'prediction';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    start=time.perf_counter();rows=merged_rows(runs)
    save_csv(out/'full_distribution.csv',rows)
    pred,scores,coefficients=leave_seed_out(rows,cfg)
    save_csv(out/'predictions.csv',pred);save_csv(out/'scores.csv',scores)
    write_json(out/'coefficients.json',coefficients)
    samples,intervals=hierarchical_bootstrap(rows,runs,cfg)
    save_csv(out/'bootstrap.csv',samples);write_json(out/'bootstrap_intervals.json',intervals)
    write_json(out/'spearman.json',stratified_spearman(rows,cfg))
    examples=quantile_examples(rows,cfg);save_csv(out/'quantile_counterexamples.csv',examples)
    result={'rows':len(rows),'held_out_primary_rows_per_seed':192,'counterexamples':len(examples),
            'bootstrap_requested':cfg['analysis']['bootstrap_replicates'],'CPU_wall_seconds':time.perf_counter()-start,
            'all_seeds_retained':cfg['seeds'],'scores':scores,'intervals':intervals}
    write_json(out/'summary.json',result)
    return result
