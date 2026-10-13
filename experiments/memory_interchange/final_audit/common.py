from pathlib import Path
import csv
import json
import time
import hashlib
from collections import Counter
import numpy as np
import torch
import yaml
from ..common import ROOT as BASE, config, write_json, sha256, seed_all
from ..truth import contents, CONDITIONS, SUFFIXES, matrix_apply, inverse

OUT = BASE / 'Final_audit'
RUNS = BASE / 'runs/v1_3'
OLD = BASE / 'results/v1_3'
SEEDS = list(range(101, 107))
MODES = ('all', 'rec', 'both', 'E_sem')
SWAP = {'NN': 'NN', 'NA': 'NB', 'NB': 'NA'}
PAIRS = [(a, b) for a in range(18) for b in range(18) if a != b]

def csv_read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))

def csv_write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def maps_answers(m):
    return np.stack((m, inverse(m)), axis=-2)

class Budget:
    def __init__(self):
        self.inherited = json.loads((OLD/'budget.json').read_text())['gpu_seconds']
        self.path = OUT/'budget.json'; self.start = None
    @property
    def used(self):
        return json.loads(self.path.read_text())['gpu_seconds'] if self.path.exists() else self.inherited
    def check(self):
        active = time.perf_counter()-self.start if self.start is not None else 0
        if self.used+active >= 43200: raise RuntimeError('12 GPU-hour limit reached')
    def __enter__(self):
        self.check(); self.start = time.perf_counter() if torch.cuda.is_available() else None
        return self
    def __exit__(self, *args):
        if self.start is not None:
            torch.cuda.synchronize(); total = self.used+time.perf_counter()-self.start
            write_json(self.path, dict(gpu_seconds=total, inherited_gpu_seconds=self.inherited,
                       final_audit_gpu_seconds=total-self.inherited, limit_seconds=43200))
        self.start = None

def preserve_check():
    plan = json.loads((OUT/'preservation_manifest.json').read_text())
    bad = [p for p, v in plan.items() if sha256(BASE/p) != v['sha256']]
    assert not bad, bad
    return {'files': len(plan), 'mismatches': bad}

def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'frozen_plan.json').exists():
        preserve_check()
        return
    prior = json.loads((OLD/'preserved_versions.json').read_text())['files']
    prior.update(json.loads((OLD/'artifact_manifest.json').read_text()))
    for p in [OLD/'artifact_manifest.json', BASE/'split_manifest.json']:
        prior[str(p.relative_to(BASE))] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    write_json(OUT/'preservation_manifest.json', prior)
    preserve_check()
    manifest = json.loads((BASE/'split_manifest.json').read_text())
    q, ids = contents(manifest, 'B'); b1, b1ids = contents(manifest, 'B', True)
    used = set(map(tuple, b1)); keep = np.array([tuple(x) not in used for x in q])
    reserved, rid = q[keep], ids[keep]
    assert set(rid) == set(manifest['reserved_behavior_orbit_ids'])
    assert len(reserved) == 362 and len(set(rid)) == 51
    for s in SEEDS:
        cache = np.load(RUNS/f'seed_{s}/analysis/behavior_states.npz')
        assert np.array_equal(cache['content'], b1) and np.array_equal(cache['orbit_id'], b1ids)
    # Evidence is the preserved local source and all stored natural state caches,
    # not a claim that unrecorded external access can be reconstructed.
    previous = []
    reserved_set = set(map(tuple, reserved))
    for path in sorted((BASE/'runs').rglob('*states.npz')):
        with np.load(path) as f:
            if 'content' not in f: continue
            overlap = sum(tuple(x) in reserved_set for x in f['content'])
            assert overlap == 0, f'Reserved contents in historical neural cache: {path}'
            previous.append({'path': str(path.relative_to(BASE)), 'contents': len(f['content']),
                             'reserved_overlap': overlap, 'sha256': sha256(path)})
    np.savez_compressed(OUT/'B_reserved_members.npz', content=reserved, orbit_id=rid,
                        B1_content=b1, B1_orbit_id=b1ids)
    write_json(OUT/'B_reserved_manifest.json', {'split_sha256': sha256(BASE/'split_manifest.json'),
        'total_B_contents': len(q), 'total_B_orbits': len(set(ids)),
        'B1_contents': len(b1), 'B1_orbits': len(set(b1ids)),
        'reserved_contents': len(reserved), 'reserved_orbits': len(set(rid)),
        'orbit_ids': sorted(map(int, set(rid))), 'members': reserved.tolist(),
        'member_orbit_ids': rid.tolist(), 'content_array_sha256': digest_array(reserved),
        'members_file_sha256': sha256(OUT/'B_reserved_members.npz'),
        'historical_state_caches': previous, 'known_prior_neural_access': False,
        'qualification': 'No prior reserved neural outputs/rule tuning in preserved local run records; manifest membership alone was previously known.'})
    checkpoints = []
    for s in SEEDS:
        summ = json.loads((RUNS/f'seed_{s}/summary.json').read_text())
        p = BASE/summ['selected_checkpoint']
        checkpoints.append({'seed': s, 'path': str(p.relative_to(BASE)), 'sha256': sha256(p),
                            'stage7_step': summ['selected_step_in_stage7']})
    bs = [r for r in csv_read(OLD/'prediction/bootstrap.csv') if r['mode']=='all' and
          r['segment']=='unseen_cross' and r['model']=='M_CKA_v13']
    median = np.median([float(r['mae_improvement']) for r in bs])
    typical = min(bs, key=lambda r:(abs(float(r['mae_improvement'])-median), int(r['replicate'])))
    lo = min(bs, key=lambda r:float(r['mae_improvement']))
    hi = max(bs, key=lambda r:float(r['mae_improvement']))
    plan = {
      'version': 'v1.3 Final audit', 'date': '2026-10-12', 'network_training': False,
      'preserved_versions': ['v1.0','v1.1','v1.2','v1.3'], 'checkpoints': checkpoints,
      'B1_status': 'EXPLORATORY: new donor-rule hypothesis on already opened B1',
      'B_reserved_status': 'independent confirmation for the frozen translation contrast',
      'reserved_manifest_sha256': sha256(OUT/'B_reserved_manifest.json'),
      'coverage': {'conditions': CONDITIONS, 'directions': 'all 306 ordered non-self pairs',
                   'delays': [0,10], 'suffixes': SUFFIXES, 'queries': 'all 7 item + 7 position'},
      'translation': SWAP, 'targets': ['F_b(S)(q)', 'F_b(swap(S))(q)'],
      'primary': 'cross-context accuracy(patch+swapS, original target) minus accuracy(patch+S, original target)',
      'aggregation': 'equal seeds; equal ordered directions (therefore equal both directions within each pair); equal d,S,query type; within each direction/mask all retained content-query observations pooled. Main all includes NN. Conditional masks use the SAME observations for both treatments.',
      'masks': {'all': 'all observations', 'rec': 'receiver native original S correct',
                'both': 'rec and donor native same visible original S correct in its own semantics',
                'E_sem': 'rec and donor native swap(S) if cross, original S if same, equals receiver original semantic target'},
      'mask_freeze': 'all natural/native outputs and mask files saved before new memory patches',
      'CF': ['F_b(q)', 'F_b(G_b(p_a))', 'F_a(q)'],
      'CF_subsets': ['all','receiver_initial_distinct','receiver_donor_distinct',
                     'initial_donor_distinct','all_three_distinct'],
      'controls': ['native original','native translated','patch original','patch translated','sham',
                   'NN exact replay','native translation literal target','same-context translation original target'],
      'block_uncertainty': {'replicates':2000, 'seed':20261018,
             'units':'resample 6 seeds and whole reserved orbits; all directions/queries retained; no RNN/probe refitting'},
      'context_probes': {'alpha':1, 'target':[-1,1], 'threshold':0,
        'fit':'T fixed 512 contents; exclude AB/BA; balanced r,H,d; c one row per unique (r,H,d)',
        'variants':['pooled d','fit d=0, test both d','fit d=10, test both d'],
        'scaling':'T fit-only coordinate mean/std; exactly zero std replaced by 1; no score selection',
        'tests':['T nonmixed descriptive','T AB/BA history holdout','G nonmixed content holdout','G AB/BA both holdouts']},
      'geometry':'G only; deduplicate (seed,a,b,d); original raw denominator epsilon=1e-12; no transformations of intervention states',
      'bootstrap_replay': {'replicate_ids':sorted({0,285,int(typical['replicate']),int(lo['replicate']),int(hi['replicate'])}),
         'typical':int(typical['replicate']), 'minimum':int(lo['replicate']), 'maximum':int(hi['replicate']),
         'models':['M0_v13','M_CKA_v13'], 'modes':['all','rec','both'],
         'original_rng_seed':20261016, 'all_held_original_seed_copies_excluded':True},
      'fixed_predictions': {'folds':6, 'mode':'all', 'segment':'unseen_cross',
           'models':['M0_v13','M_CKA_v13'], 'replicates':100000, 'seed':20261017,
           'conditional_only':True, 'no_refit':True},
      'budget': {'inherited_gpu_seconds':json.loads((OLD/'budget.json').read_text())['gpu_seconds'], 'limit_seconds':43200},
      'stop':'finish these three audits; no models/adapters/parameter scans/ablations/four-step extension/publishing'
    }
    (OUT/'revision_final_audit.yaml').write_text(yaml.safe_dump(plan, allow_unicode=True, sort_keys=False))
    write_json(OUT/'frozen_plan.json', plan)
    write_json(OUT/'freeze_receipt.json', {'unix_time':time.time(), 'before_new_neural_outputs':True,
               'config_sha256':sha256(OUT/'revision_final_audit.yaml'),
               'plan_sha256':sha256(OUT/'frozen_plan.json'), 'preservation':preserve_check()})
    print(json.dumps({'frozen':True,'reserved_contents':len(reserved),'reserved_orbits':len(set(rid))}),flush=True)

if __name__=='__main__': freeze()
