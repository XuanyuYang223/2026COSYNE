#!/usr/bin/env python3
"""Read-only rescoring of frozen B1 wrong-content interventions. No torch import."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import yaml

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
REPO = BASE.parent.parent
SEEDS = tuple(range(101, 107))
DELAYS = (0, 10)
SUFFIXES = ('NN', 'NA', 'NB')
hist_node = next(n for n in ast.parse((BASE/'truth.py').read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HISTORIES' for t in n.targets))
HISTORIES = ast.literal_eval(hist_node.value)
CONDITIONS = tuple((r, h) for r in (0, 1) for h in HISTORIES)
CONTEXT = np.array([r for r, h in CONDITIONS])
HISTORY = np.array([h for r, h in CONDITIONS])
CROSS = CONTEXT[:, None] != CONTEXT[None, :]
SAME_H = HISTORY[:, None] == HISTORY[None, :]
COUNT_METRICS = (
    'donor_math', 'receiver_original_math', 'recombined_math', 'other',
    'reproduce_donor_native', 'donor_native_correct', 'receiver_native_correct',
    'donor_correct_reproduced', 'donor_correct_not_reproduced',
    'donor_error_reproduced', 'donor_error_not_reproduced',
    'not_donor_math_reproduced_donor_error', 'donor_error_patch_donor_truth',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csv_write(name, rows):
    import csv
    rows = list(rows)
    if not rows:
        (OUT/name).write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: ('null' if v is None else v) for k, v in row.items()})


def headers(path):
    result = {}
    with zipfile.ZipFile(path) as z:
        for key in z.namelist():
            with z.open(key) as f:
                version = np.lib.format.read_magic(f)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}[version]
                shape, order, dtype = reader(f)
            result[key[:-4]] = {'shape': list(shape), 'dtype': str(dtype), 'fortran': order}
    return result


def apply_array(q, word, r):
    q = np.array(q, dtype=np.int64, copy=True)
    for cue in word:
        if cue == 'N':
            continue
        q = np.argsort(q, axis=-1) if (cue == 'A') == (r == 0) else 6-q
    return q


def truth_arrays(q, suffix):
    forward = np.array([apply_array(q, suffix, r) for r, _ in CONDITIONS])
    return np.stack((forward, np.argsort(forward, axis=-1)), axis=2)


def scalar_answer(q, suffix, context, kind, item):
    # Independent list-based inverse, not the vectorized argsort implementation.
    p = list(map(int, q))
    for cue in suffix:
        if cue == 'N':
            continue
        if (cue == 'A') == (context == 0):
            inverse = [None]*7
            for i, position in enumerate(p):
                inverse[position] = i
            p = inverse
        else:
            p = [6-position for position in p]
    return p[item] if kind == 0 else p.index(item)


def matrix_truth(q, suffix, context):
    matrix = np.eye(7, dtype=np.int64)[q]
    for cue in suffix:
        if cue != 'N':
            matrix = matrix.swapaxes(-1, -2) if (cue == 'A') == (context == 0) else matrix[..., ::-1]
    return matrix.argmax(-1)


def canonical_contents():
    manifest = read_json(BASE/'split_manifest.json')
    q, ids = [], []
    for oid in manifest['behavior_orbit_ids']:
        q.extend(manifest['orbits'][oid]['members'])
        ids.extend([oid]*len(manifest['orbits'][oid]['members']))
    assert len(q) == 134 and len(set(ids)) == 19
    return manifest, np.array(q), np.array(ids)


def state_metadata(seed, q, ids):
    path = BASE/f'runs/v1_3/seed_{seed}/analysis/behavior_states.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z['content'], q) and np.array_equal(z['orbit_id'], ids)
        assert np.array_equal(z['contexts'], CONTEXT) and np.array_equal(z['histories'], HISTORY)
        assert np.array_equal(z['delays'], DELAYS)
        assert np.array_equal(z['cache_steps'], [21, 31])
        initial = z['initial']
        assert initial.shape == (2, 18, 134, 7)
        trajectory = z['truth_trajectory']
        for di in range(2):
            for a, (r, h) in enumerate(CONDITIONS):
                assert np.array_equal(initial[di, a], apply_array(q, h[::-1], r))
                assert np.array_equal(trajectory[di, a, :, 0], initial[di, a])
                assert np.array_equal(trajectory[di, a, :, 1], apply_array(initial[di, a], h[:1], r))
                assert np.array_equal(trajectory[di, a, :, 2], q)
        control = z['control']
        assert control.shape == (2, 18, 32) and np.isfinite(control).all()
        return {'control': control, 'initial': initial}


def mismatches(seed, q, ids, mismatch_seed):
    with np.load(BASE/f'runs/v1_3/seed_{seed}/analysis/mismatch.npz', allow_pickle=False) as z:
        idx = z['indices']
        assert np.array_equal(np.sort(idx), np.arange(134))
        assert np.all(ids[idx] != ids) and np.all(np.any(q[idx] != q, axis=1))
        assert np.array_equal(z['recipient'], q) and np.array_equal(z['donor'], q[idx])
        assert np.array_equal(z['orbit_id'], ids)
    rng = np.random.default_rng(mismatch_seed)
    for _ in range(10000):
        expected = rng.permutation(134)
        if np.all(ids[expected] != ids):
            break
    assert np.array_equal(idx, expected)
    return idx


def suffix_schedules(cfg):
    # Evaluate only the actual pure input-encoding functions, never a model.
    tree = ast.parse((BASE/'task.py').read_text())
    names = {'blank_inputs', 'phase_at', 'suffix_inputs'}
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    constants = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Tuple))
    scope = {'np': np}
    exec(compile(ast.Module(body=[constants, *definitions], type_ignores=[]), 'task_input_functions', 'exec'), scope)
    previous = read_json(BASE/'donor_consistency/analysis_plan.json')['input_schedules']
    rows = []
    for suffix in SUFFIXES:
        x = scope['suffix_inputs']([suffix]*14, np.repeat([0, 1], 7), np.tile(np.arange(7), 2), cfg)
        hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in x.items()}
        old = next(r for r in previous if r['effective_suffix'] == suffix)
        assert hashes == old['input_sha256']
        assert not x['ctrl'][:, :, :2].any() and not x['stim'].any()
        assert np.all(x['lengths'] == 16)
        rows.append({**old, 'reconstructed_from_actual_input_functions': True,
                     'absolute_swap_steps': [21, 31], 'first_query_steps': [35, 45],
                     'final_readout_steps': [37, 47]})
    return rows


def control_pairs(control, di):
    if control is None:
        return np.full((18, 18), 'missing', dtype='<U7'), np.zeros((18, 18))
    c = control[di]
    equal = (c[:, None] == c[None, :]).all(-1)
    distance = np.linalg.norm(c[:, None].astype(np.float64)-c[None, :].astype(np.float64), axis=-1)
    return np.where(equal, 'equal', 'unequal'), distance


def prepare():
    assert not (OUT/'analysis_plan.json').exists(), 'Existing plan is immutable; use score/verify.'
    manifest = read_json(BASE/'results/v1_3/artifact_manifest.json')
    cfg = yaml.safe_load((BASE/'config.yaml').read_text())
    split, q, ids = canonical_contents()
    required = []
    for seed in SEEDS:
        prefix = f'runs/v1_3/seed_{seed}'
        required += [(f'{prefix}/selected_stage7.pt', True), (f'{prefix}/summary.json', True),
                     (f'{prefix}/analysis/mismatch.npz', True),
                     (f'{prefix}/analysis/behavior_states.npz', False),
                     (f'{prefix}/analysis/baseline_controls.json', True)]
        for d, suffix in itertools.product(DELAYS, SUFFIXES):
            required += [(f'{prefix}/analysis/raw_d{d}_S{suffix}.npz', True),
                         (f'{prefix}/analysis/baseline_d{d}_S{suffix}.npz', True)]
    required += [(p, True) for p in ['config.yaml', 'split_manifest.json', 'truth.py', 'task.py',
        'model.py', 'cache.py', 'intervene.py', 'v1_3/intervene.py', 'v1_3/controls.py',
        'v1_3/science.py', 'results/v1_3/report.md', 'results/v1_3/artifact_manifest.json',
        'results/v1_3/frozen_plan.json', 'donor_consistency/report.md', 'donor_consistency/analysis_plan.json']]
    records, errors = [], []
    for relative, critical in required:
        path = BASE/relative
        row = {'path': relative, 'critical': critical, 'exists': path.exists()}
        if path.exists():
            row.update(bytes=path.stat().st_size, sha256=sha(path))
            row['original_manifest_sha256'] = manifest.get(relative, {}).get('sha256')
            if row['original_manifest_sha256']:
                row['manifest_matches'] = row['sha256'] == row['original_manifest_sha256']
                if not row['manifest_matches']:
                    errors.append({'path': relative, 'reason': 'hash_mismatch'})
            if path.suffix == '.npz':
                row['headers_only'] = headers(path)
            if '/raw_d' in relative:
                h = row['headers_only']
                for key, shape in [('wrong_logp', [18,18,134,2,7,7]), ('matched_logp', [18,18,134,2,7,7]),
                                   ('native_logp', [18,134,2,7,7]), ('answers', [18,134,2,7]),
                                   ('counterfactual_answers', [18,134,2,7])]:
                    assert h[key]['shape'] == shape, (relative, key, h[key])
                assert row['original_manifest_sha256'], 'Raw source must be registered.'
        elif critical:
            errors.append({'path': relative, 'reason': 'missing_primary_input'})
        records.append(row)
    write_json('input_inventory.json', {'records': records, 'blocking_errors': errors,
                                      'output_values_read': False})
    if errors:
        write_json('status.json', {'status': 'BLOCKED', 'errors': errors})
        (OUT/'report.md').write_text('# Wrong-content 补充分析：BLOCKED\n\n主缓存缺失或散列无法核验；未评分、未运行网络。详见 input_inventory.json。\n')
        return
    checkpoints, samples, controls = [], [], []
    old_checkpoints = {r['seed']:r for r in read_json(BASE/'donor_consistency/analysis_plan.json')['checkpoints']}
    for seed in SEEDS:
        summary = read_json(BASE/f'runs/v1_3/seed_{seed}/summary.json')
        assert summary['seed'] == seed and summary['selected_checkpoint'] == f'runs/v1_3/seed_{seed}/selected_stage7.pt'
        checkpoints.append({'seed': seed, 'path': summary['selected_checkpoint'],
                            'sha256': sha(BASE/summary['selected_checkpoint']),
                            'stage7_step': summary['selected_step_in_stage7']})
        assert checkpoints[-1]['sha256']==old_checkpoints[seed]['sha256']
        assert checkpoints[-1]['stage7_step']==old_checkpoints[seed]['stage7_step']
        state = state_metadata(seed, q, ids)
        idx = mismatches(seed, q, ids, cfg['analysis']['mismatch_seed'])
        controls.append({'seed': seed, 'available': state is not None})
        for di, d in enumerate(DELAYS):
            relation, _ = control_pairs(None if state is None else state['control'], di)
            for suffix in SUFFIXES:
                answers = truth_arrays(q, suffix)
                for kind, rdon, same_h, crel, selection in itertools.product(
                        (0,1), (0,1), (True,False), ('equal','unequal','missing'), ('all','triple_distinct')):
                    possible = CROSS & (CONTEXT[:,None] == rdon) & (SAME_H == same_h) & (relation == crel)
                    chosen = None
                    for donor, receiver in np.argwhere(possible):
                        a = answers[donor, idx, kind]
                        b = answers[receiver, :, kind]
                        c = answers[receiver, idx, kind]
                        eligible = (a != b) & (a != c) & (b != c) if selection == 'triple_distinct' else np.ones_like(a, bool)
                        if eligible.any():
                            j, item = np.argwhere(eligible)[0]
                            chosen = {'seed': seed, 'd': d, 'S': suffix, 'D': int(donor), 'R': int(receiver),
                                      'donor_condition': list(CONDITIONS[donor]), 'receiver_condition': list(CONDITIONS[receiver]),
                                      'j': int(j), 'k': int(idx[j]), 'query_type': kind, 'query_item': int(item),
                                      'q_j': q[j].tolist(), 'q_k': q[idx[j]].tolist(), 'selection': selection,
                                      'history_relation': 'same' if same_h else 'different', 'control_relation': crel}
                            break
                    if chosen:
                        samples.append(chosen)
    write_json('fixed_direct_sample.json', samples)
    tracked = subprocess.check_output(['git','ls-files','-z'], cwd=REPO).decode().rstrip('\0').split('\0')
    before = {p: sha(REPO/p) for p in tracked if (REPO/p).is_file()}
    for row in records:
        if row['exists']:
            before[str((BASE/row['path']).relative_to(REPO))] = row['sha256']
    write_json('preservation_before.json', before)
    plan = {
        'date': '2026-10-17', 'timezone': 'America/Los_Angeles',
        'role': 'Descriptive rescoring of already-opened frozen v1.3/B1 wrong-content outputs; not independent confirmation.',
        'freeze_scope': 'This analysis only; saved before reading any wrong/matched/native probability values for this task. Not retrospective preregistration.',
        'AGENTS': 'No applicable AGENTS.md found in repository or ancestors.',
        'seeds': list(SEEDS), 'dataset': 'v1.3/B1', 'contents': 134, 'orbits': 19,
        'B_reserved': 'Excluded: no corresponding wrong-content cache is used.',
        'content': q.tolist(), 'orbit_id': ids.tolist(),
        'conditions': [{'id': i, 'r': r, 'H': h} for i,(r,h) in enumerate(CONDITIONS)],
        'delays': list(DELAYS), 'suffixes': list(SUFFIXES), 'class_order': list(range(7)),
        'query_types': {'0':'item_to_position', '1':'position_to_item'},
        'axes': {'wrong_logp':'D,R,j,type,item,class', 'matched_logp':'D,R,content,type,item,class',
                 'native_logp':'condition,content,type,item,class', 'answers':'condition,content,type,item',
                 'control':'delay,condition,control_neuron; NO content axis'},
        'pairing_key': ['dataset','seed','D','R','q_j','q_k','d','actual_suffix_S','query_type','query_item','swap_event'],
        'candidates': {'donor_math':'answers[D,mismatch_index[j],type,item]',
                       'receiver_original_math':'answers[R,j,type,item]',
                       'recombined_math':'counterfactual_answers[R,j,type,item] == answers[R,mismatch_index[j],type,item]',
                       'donor_native':'argmax(native_logp[D,mismatch_index[j],type,item,:])',
                       'receiver_native':'argmax(native_logp[R,j,type,item,:])'},
        'main_selection': 'Cross-context D,R; triple_distinct iff the THREE MATHEMATICAL answers are pairwise distinct. No prediction/correctness/native-agreement selection.',
        'all_selection': 'All cross-context queries, including overlapping mathematical candidates.',
        'not_three_mask': 'Existing v1.3 three is native-correctness selection and is never used here.',
        'main_categories': ['donor_math','receiver_original_math','recombined_math','other'],
        'cross_directions': [{'D': int(a), 'R': int(b)} for a,b in np.argwhere(CROSS)],
        'argmax': 'Lowest class index on ties, applied to saved log probabilities.',
        'aggregation': 'Pool eligible query numerators/denominators within each seed, then equally average six seed rates. Expanded pooled counts/rates separate. No direction reweighting after eligibility.',
        'empty_strata': 'n=0, rate=null. NN triple_distinct stays empty. No imputation. Strict six-seed means null if any seed stratum empty; available-seed means separately labeled.',
        'strata': ['d','S','query_type','same/different history','exactly equal/unequal/missing c','donor context direction', 'full d*S*query_type*history*c crossing'],
        'control_equality': 'Elementwise exact equality in original stored coordinates; L2 computed after float64 conversion, with no threshold or content axis.',
        'donor_native_unique_key': ['seed','D','q_k','d','S','query_type','query_item'],
        'reuse': 'Keep donor-query multiplicity counts; recipients are not independent native predictions.',
        'wrong_matched_consistency': 'wrong[D,R,j] versus matched[D,R,mismatch_index[j]] for all 324 stored directions, plus cross-context separately; diagonal controls excluded from science.',
        'probabilities': 'float64 exp(logp), finite-value and row-sum checks, then row normalization; no logit comparison or clipping.',
        'raw_normalization_atol': 2e-6, 'normalized_atol': 3e-15,
        'independent_probability_reduction_atol': 3e-15,
        'descriptive_probability_tolerances': [1e-6,1e-5,1e-4],
        'precision_note': 'Original GPU controls passed; original strict CPU float32 logit tolerance failed; identical float64 coefficients agreed and original compared argmax unchanged. New probability tolerances do not replace that failure.',
        'statistics': 'Descriptive only; no confidence intervals, p values, bootstrap or refitting.',
        'input_schedules': suffix_schedules(cfg), 'checkpoints': checkpoints, 'control_availability': controls,
        'input_inventory_sha256': sha(OUT/'input_inventory.json'),
        'fixed_direct_sample_sha256': sha(OUT/'fixed_direct_sample.json'),
        'direct_sample_rule': 'Before outputs, choose lexicographically first cross pair and first j/item per seed,d,S,type,donor context,history,c stratum and selection using mathematical candidates only.',
        'no_network_forward': True, 'no_training': True, 'no_new_models': True,
        'missing_policy': 'Any missing/unverifiable primary seed cache blocks six-model scoring; optional c absence is explicit.',
        'code_sha256': {'analysis.py': sha(Path(__file__))},
    }
    write_json('analysis_plan.json', plan)
    write_json('freeze_receipt.json', {'analysis_plan_sha256': sha(OUT/'analysis_plan.json'),
                                     'created_utc': datetime.now(timezone.utc).isoformat(),
                                     'scored_output_values_read': False})
    print(json.dumps({'prepared': True, 'raw_caches': 36, 'seeds': list(SEEDS),
                      'direct_samples': len(samples), 'plan_sha256': sha(OUT/'analysis_plan.json')}), flush=True)


def load_plan():
    plan = read_json(OUT/'analysis_plan.json')
    assert sha(OUT/'analysis_plan.json') == read_json(OUT/'freeze_receipt.json')['analysis_plan_sha256']
    assert sha(Path(__file__)) == plan['code_sha256']['analysis.py'], 'Frozen scoring source changed.'
    assert sha(OUT/'fixed_direct_sample.json') == plan['fixed_direct_sample_sha256']
    assert sha(OUT/'input_inventory.json') == plan['input_inventory_sha256']
    return plan


def groups():
    result = [('overall','all',{})]
    for field, values in [('d',DELAYS),('S',SUFFIXES),('query_type',(0,1)),
                          ('history_relation',('same','different')),('control_relation',('equal','unequal','missing')),
                          ('donor_context',(0,1))]:
        result += [(field,str(v),{field:v}) for v in values]
    for d,s,t,h,c in itertools.product(DELAYS,SUFFIXES,(0,1),('same','different'),('equal','unequal','missing')):
        result.append(('full',f'd={d};S={s};type={t};history={h};c={c}',
                       {'d':d,'S':s,'query_type':t,'history_relation':h,'control_relation':c}))
    return result


def empty_counts():
    return {'base_n':0, 'selected_n':0, 'unique_donor_queries':0,
            'control_observed_queries':0, 'control_L2_weighted_sum':0., 'control_L2_min':None, 'control_L2_max':None,
            **{k+'_n':0 for k in COUNT_METRICS}}


def add_counts(total, row):
    for k,v in row.items():
        if k == 'control_L2_min':
            if v is not None:
                total[k] = v if total[k] is None else min(total[k], v)
        elif k == 'control_L2_max':
            if v is not None:
                total[k] = v if total[k] is None else max(total[k], v)
        else:
            total[k] += v


def rate_row(count):
    n = count['selected_n']
    return {**count, 'coverage': n/count['base_n'] if count['base_n'] else None,
            **{k+'_rate': count[k+'_n']/n if n else None for k in COUNT_METRICS},
            'control_L2_query_weighted_mean': count['control_L2_weighted_sum']/count['control_observed_queries']
                                            if count['control_observed_queries'] else None}


def probabilities(logp):
    assert np.isfinite(logp).all(), 'Nonfinite saved output.'
    p = np.exp(logp.astype(np.float64))
    sums = p.sum(-1, keepdims=True)
    error = float(np.max(np.abs(sums-1)))
    assert error <= 2e-6, ('Original probability normalization failed.',error)
    p /= sums
    assert np.max(np.abs(p.sum(-1)-1)) <= 3e-15
    return p, error


def probability_summary(maximum, tv, agreement):
    result = {'n': int(maximum.size), 'answers_agree_n':int(agreement.sum()),
              'answer_agreement':float(agreement.mean())}
    for name, data in [('max_abs_probability_difference',maximum),('TV',tv)]:
        result[name] = {'mean':float(data.mean()),'median':float(np.median(data)),
                        'q95':float(np.quantile(data,.95)), 'q99':float(np.quantile(data,.99)),
                        'max':float(data.max())}
    result['close_counts'] = {str(t):int((maximum<=t).sum()) for t in (1e-6,1e-5,1e-4)}
    result['close_rates'] = {k:v/maximum.size for k,v in result['close_counts'].items()}
    result['exact_probability_vector_n'] = int((maximum==0).sum())
    return result


def score():
    started = time.perf_counter()
    plan = load_plan()
    inventory = read_json(OUT/'input_inventory.json')['records']
    for record in inventory:
        if record['exists']:
            assert sha(BASE/record['path']) == record['sha256'], record['path']
    cfg = yaml.safe_load((BASE/'config.yaml').read_text())
    _,q,ids = canonical_contents()
    group_list = groups()
    aggregate = {(s,f,l,selection):empty_counts() for s in SEEDS for f,l,_ in group_list
                 for selection in ('all','triple_distinct')}
    samples = read_json(OUT/'fixed_direct_sample.json')
    sample_cells = defaultdict(list)
    for sample in samples:
        sample_cells[sample['seed'],sample['d'],sample['S']].append(sample)
    index_rows, direct_rows, relation_rows, control_rows, reuse_rows = [],[],[],[],[]
    disagreements, native_exceptions, overlap_rows, normalization = [],[],[],[]
    probability_buffers = {'all_stored_directions': [[],[],[]], 'cross_context': [[],[],[]]}
    for seed in SEEDS:
        state = state_metadata(seed,q,ids)
        registered_controls = read_json(BASE/f'runs/v1_3/seed_{seed}/analysis/baseline_controls.json')
        assert registered_controls['passed']
        idx = mismatches(seed,q,ids,cfg['analysis']['mismatch_seed'])
        usage = {selection:np.zeros((2,3,18,134,2,7),np.uint8) for selection in ('all','triple_distinct')}
        for di,d in enumerate(DELAYS):
            relation, distance = control_pairs(None if state is None else state['control'],di)
            for donor,receiver in np.argwhere(CROSS):
                control_rows.append({'seed':seed,'d':d,'D':int(donor),'R':int(receiver),
                    'H_D':CONDITIONS[donor][1],'H_R':CONDITIONS[receiver][1],
                    'history_relation':'same' if SAME_H[donor,receiver] else 'different',
                    'control_relation':relation[donor,receiver],
                    'control_L2':float(distance[donor,receiver]) if state else None})
            for si,suffix in enumerate(SUFFIXES):
                raw_path = BASE/f'runs/v1_3/seed_{seed}/analysis/raw_d{d}_S{suffix}.npz'
                with np.load(raw_path,allow_pickle=False) as raw:
                    assert int(raw['seed'])==seed and int(raw['d'])==d and str(raw['S'])==suffix
                    assert np.array_equal(raw['content'],q) and np.array_equal(raw['orbit_id'],ids)
                    assert np.array_equal(raw['mismatch_index'],idx)
                    if state:
                        assert np.array_equal(raw['initial'],state['initial'][di])
                    answers = raw['answers']
                    assert np.array_equal(answers,truth_arrays(q,suffix))
                    for a,(r,h) in enumerate(CONDITIONS):
                        assert np.array_equal(answers[a,:,0],matrix_truth(q,suffix,r))
                    cf = raw['counterfactual_answers']
                    assert np.array_equal(cf,answers[:,idx])
                    assert np.array_equal(raw['current_future'],answers[:,:,0])
                    assert np.array_equal(raw['wrong_final_content'],cf[:,:,0])
                    wrong = raw['wrong_logp']; matched = raw['matched_logp']; native = raw['native_logp']
                    assert wrong.shape == matched.shape == (18,18,134,2,7,7)
                    with np.load(raw_path.parent/f'baseline_d{d}_S{suffix}.npz',allow_pickle=False) as baseline:
                        assert np.array_equal(native,baseline['native_logp'])
                    prediction = wrong.argmax(-1)
                    native_prediction = native.argmax(-1)
                    donor_math = np.broadcast_to(answers[:,idx][:,None],prediction.shape)
                    receiver_math = np.broadcast_to(answers[None],prediction.shape)
                    recombined = np.broadcast_to(cf[None],prediction.shape)
                    donor_prediction = np.broadcast_to(native_prediction[:,idx][:,None],prediction.shape)
                    recipient_prediction = np.broadcast_to(native_prediction[None],prediction.shape)
                    triple = (donor_math!=receiver_math)&(donor_math!=recombined)&(receiver_math!=recombined)
                    if suffix=='NN':
                        assert not triple.any()
                    cross = np.broadcast_to(CROSS[:,:,None,None,None],prediction.shape)
                    donor_correct = donor_prediction == donor_math
                    reproduced = prediction == donor_prediction
                    values = {'donor_math':prediction==donor_math, 'receiver_original_math':prediction==receiver_math,
                              'recombined_math':prediction==recombined, 'reproduce_donor_native':reproduced,
                              'donor_native_correct':donor_correct, 'receiver_native_correct':recipient_prediction==receiver_math,
                              'donor_correct_reproduced':donor_correct&reproduced,
                              'donor_correct_not_reproduced':donor_correct&~reproduced,
                              'donor_error_reproduced':~donor_correct&reproduced,
                              'donor_error_not_reproduced':~donor_correct&~reproduced,
                              'not_donor_math_reproduced_donor_error':(prediction!=donor_math)&~donor_correct&reproduced,
                              'donor_error_patch_donor_truth':~donor_correct&(prediction==donor_math)}
                    values['other'] = ~(values['donor_math']|values['receiver_original_math']|values['recombined_math'])
                    for selection,mask in [('all',cross),('triple_distinct',cross&triple)]:
                        counts_j = mask.sum(axis=1)
                        usage[selection][di,si] = counts_j[:,np.argsort(idx)]
                        assert int(usage[selection][di,si].sum())==int(mask.sum())
                    overlap_rows.append({'seed':seed,'d':d,'S':suffix,'cross_all_n':int(cross.sum()),
                        'triple_distinct_n':int((cross&triple).sum()),
                        'donor_receiver_coincident_n':int((cross&(donor_math==receiver_math)).sum()),
                        'donor_recombined_coincident_n':int((cross&(donor_math==recombined)).sum()),
                        'receiver_recombined_coincident_n':int((cross&(receiver_math==recombined)).sum())})
                    for family,level,filters in group_list:
                        if ('d'in filters and filters['d']!=d) or ('S'in filters and filters['S']!=suffix):
                            continue
                        pair_keep = CROSS.copy()
                        if 'history_relation'in filters:
                            pair_keep &= SAME_H if filters['history_relation']=='same' else ~SAME_H
                        if 'control_relation'in filters:
                            pair_keep &= relation==filters['control_relation']
                        if 'donor_context'in filters:
                            pair_keep &= CONTEXT[:,None]==filters['donor_context']
                        base = np.broadcast_to(pair_keep[:,:,None,None,None],prediction.shape).copy()
                        if 'query_type'in filters:
                            base[:,:,:,1-filters['query_type'],:] = False
                        for selection in ('all','triple_distinct'):
                            mask = base if selection=='all' else base&triple
                            row = empty_counts();row['base_n']=int(base.sum());row['selected_n']=int(mask.sum())
                            row['unique_donor_queries']=int(np.count_nonzero(mask.sum(axis=1)))
                            row.update({key+'_n':int((value&mask).sum()) for key,value in values.items()})
                            if selection=='triple_distinct':
                                assert sum(row[k+'_n'] for k in ('donor_math','receiver_original_math','recombined_math','other'))==row['selected_n']
                            assert sum(row[k+'_n'] for k in ('donor_correct_reproduced','donor_correct_not_reproduced',
                                                            'donor_error_reproduced','donor_error_not_reproduced'))==row['selected_n']
                            assert row['not_donor_math_reproduced_donor_error_n']==row['donor_error_reproduced_n']
                            if state and mask.any():
                                pair_counts = mask.sum(axis=(2,3,4))
                                row.update(control_observed_queries=row['selected_n'],
                                           control_L2_weighted_sum=float((distance*pair_counts).sum()),
                                           control_L2_min=float(distance[pair_counts>0].min()),
                                           control_L2_max=float(distance[pair_counts>0].max()))
                            add_counts(aggregate[seed,family,level,selection],row)
                    pw, ew = probabilities(wrong)
                    pm, em = probabilities(matched)
                    pn, en = probabilities(native)
                    pm = pm[:,:,idx]
                    maxdiff = np.max(np.abs(pw-pm),axis=-1)
                    tv = .5*np.sum(np.abs(pw-pm),axis=-1)
                    agree = prediction == matched.argmax(-1)[:,:,idx]
                    normalization.append({'seed':seed,'d':d,'S':suffix,'wrong_raw_sum_error':ew,
                                          'matched_raw_sum_error':em,'native_raw_sum_error':en,
                                          'wrong_ties':int(((wrong==wrong.max(-1,keepdims=True)).sum(-1)>1).sum()),
                                          'native_ties':int(((native==native.max(-1,keepdims=True)).sum(-1)>1).sum())})
                    for label,keep in [('all_stored_directions',np.ones_like(cross)),('cross_context',cross)]:
                        x,y,z = maxdiff[keep],tv[keep],agree[keep]
                        for buffer, data in zip(probability_buffers[label],(x,y,z)):
                            buffer.append(data)
                        stats = probability_summary(x,y,z)
                        relation_rows.append({'seed':seed,'d':d,'S':suffix,'scope':label,
                            'n':stats['n'],'agree_n':stats['answers_agree_n'],'agreement':stats['answer_agreement'],
                            'mean_max_abs':stats['max_abs_probability_difference']['mean'],
                            'max_max_abs':stats['max_abs_probability_difference']['max'],
                            'mean_TV':stats['TV']['mean'],'max_TV':stats['TV']['max'],
                            **{'close_'+k:v for k,v in stats['close_counts'].items()}})
                    for donor,receiver,j,kind,item in np.argwhere(~agree):
                        disagreements.append({'seed':seed,'d':d,'S':suffix,'D':int(donor),'R':int(receiver),
                            'j':int(j),'k':int(idx[j]),'query_type':int(kind),'query_item':int(item),
                            'cross_context':bool(CROSS[donor,receiver]),'wrong_answer':int(prediction[donor,receiver,j,kind,item]),
                            'matched_reindexed_answer':int(matched[donor,receiver,idx[j],kind,item].argmax()),
                            'max_abs_probability_difference':float(maxdiff[donor,receiver,j,kind,item]),
                            'TV':float(tv[donor,receiver,j,kind,item])})
                    for donor,receiver,j,kind,item in np.argwhere(cross&triple&~reproduced):
                        native_exceptions.append({'seed':seed,'d':d,'S':suffix,'D':int(donor),'R':int(receiver),
                            'j':int(j),'k':int(idx[j]),'query_type':int(kind),'query_item':int(item),
                            'wrong_answer':int(prediction[donor,receiver,j,kind,item]),
                            'donor_native_answer':int(donor_prediction[donor,receiver,j,kind,item]),
                            'donor_math':int(donor_math[donor,receiver,j,kind,item]),
                            'receiver_math':int(receiver_math[donor,receiver,j,kind,item]),
                            'recombined_math':int(recombined[donor,receiver,j,kind,item]),
                            'control_relation':relation[donor,receiver]})
                    # One independent ZIP-member read per cell, then semantic scalar indexing.
                    with zipfile.ZipFile(raw_path) as archive:
                        with archive.open('wrong_logp.npy') as member:
                            direct_wrong=np.lib.format.read_array(member,allow_pickle=False)
                        with archive.open('native_logp.npy') as member:
                            direct_native=np.lib.format.read_array(member,allow_pickle=False)
                    for sample in sample_cells[seed,d,suffix]:
                        donor = CONDITIONS.index(tuple(sample['donor_condition']))
                        receiver = CONDITIONS.index(tuple(sample['receiver_condition']))
                        j = next(i for i,p in enumerate(raw['content'].tolist()) if p==sample['q_j'])
                        k = next(i for i,p in enumerate(raw['content'].tolist()) if p==sample['q_k'])
                        kind,item = sample['query_type'],sample['query_item']
                        assert (donor,receiver,j,k)==(sample['D'],sample['R'],sample['j'],sample['k'])
                        assert k==int(raw['mismatch_index'][j])
                        direct_candidates = [scalar_answer(sample['q_k'],suffix,CONDITIONS[donor][0],kind,item),
                            scalar_answer(sample['q_j'],suffix,CONDITIONS[receiver][0],kind,item),
                            scalar_answer(sample['q_k'],suffix,CONDITIONS[receiver][0],kind,item)]
                        key=(donor,receiver,j,kind,item)
                        assert direct_candidates==[int(donor_math[key]),int(receiver_math[key]),int(recombined[key])]
                        answer = max(range(7),key=lambda c:float(direct_wrong[donor,receiver,j,kind,item,c]))
                        actual_donor = max(range(7),key=lambda c:float(direct_native[donor,k,kind,item,c]))
                        assert answer==int(prediction[key]) and actual_donor==int(donor_prediction[key])
                        scalar_prob = [math.exp(float(v)) for v in direct_wrong[donor,receiver,j,kind,item]]
                        norm = math.fsum(scalar_prob);scalar_prob = [v/norm for v in scalar_prob]
                        error = max(abs(scalar_prob[c]-pw[key][c]) for c in range(7))
                        assert error<=3e-15
                        assert sample['selection']!='triple_distinct' or len(set(direct_candidates))==3
                        direct_rows.append({**sample,'candidate_answers':direct_candidates,'prediction':answer,
                            'donor_native_prediction':actual_donor,'probability_reduction_error':error,'passed':True})
                    index_rows.append({'seed':seed,'d':d,'S':suffix,'raw_path':str(raw_path.relative_to(BASE)),
                        'metadata_checked':True,'content_order_checked':True,'condition_order_checked':True,
                        'mismatch_cross_orbit_bijection':True,'counterfactual_equals_answers_R_k':True,
                        'independent_mathematical_truth':True,'baseline_native_identical':True,
                        'future_inputs_and_readout_match':True,'cross_queries':int(cross.sum()),
                        'triple_distinct_queries':int((cross&triple).sum())})
                print(json.dumps({'scored_seed':seed,'d':d,'S':suffix}),flush=True)
        np.savez_compressed(OUT/f'donor_query_usage_seed_{seed}.npz',**usage,delays=DELAYS,suffixes=SUFFIXES,
                            content=q,orbit_id=ids,axes=np.array(['d','S','D','donor_content_k','query_type','query_item']))
        for selection,counts in usage.items():
            for reuse,number in enumerate(np.bincount(counts.ravel(),minlength=10)):
                reuse_rows.append({'seed':seed,'selection':selection,'receivers_per_native_key':reuse,
                                   'unique_native_keys':int(number),'expanded_query_records':int(reuse*number)})
            overall = aggregate[seed,'overall','all',selection]
            assert overall['unique_donor_queries']==int(np.count_nonzero(counts))
    rows = [{'seed':s,'family':f,'stratum':l,'selection':selection,**rate_row(count)}
            for (s,f,l,selection),count in aggregate.items()]
    csv_write('seed_summary.csv',[row for row in rows if row['family']=='overall'])
    csv_write('strata_summary.csv',[row for row in rows if row['family']!='overall'])
    overall_rows = []
    for family,level,_ in group_list:
        for selection in ('all','triple_distinct'):
            items=[rate_row(aggregate[s,family,level,selection]) for s in SEEDS]
            pooled=empty_counts()
            for s in SEEDS:
                add_counts(pooled,aggregate[s,family,level,selection])
            row={'family':family,'stratum':level,'selection':selection,'n_seeds':6,
                 'n_seeds_nonempty':sum(r['selected_n']>0 for r in items),**rate_row(pooled)}
            for metric in [k+'_rate' for k in COUNT_METRICS]+['coverage']:
                nonempty=[r[metric] for r in items if r[metric] is not None]
                row['six_seed_equal_mean_'+metric]=float(np.mean(nonempty)) if len(nonempty)==6 else None
                row['available_seed_mean_'+metric]=float(np.mean(nonempty)) if nonempty else None
            overall_rows.append(row)
    csv_write('overall.csv',overall_rows)
    csv_write('control_state_pairs.csv',control_rows)
    csv_write('donor_query_reuse.csv',reuse_rows)
    csv_write('candidate_overlaps.csv',overlap_rows)
    csv_write('index_checks.csv',index_rows)
    csv_write('wrong_matched_consistency.csv',relation_rows)
    csv_write('consistency_disagreements.csv',disagreements)
    csv_write('donor_native_disagreements_on_primary.csv',native_exceptions)
    write_json('direct_index_checks.json',{'n':len(direct_rows),'all_passed':all(r['passed'] for r in direct_rows),
                                         'max_probability_reduction_error':max(r['probability_reduction_error'] for r in direct_rows),
                                         'records':direct_rows})
    write_json('normalization_checks.json',normalization)
    consistency={label:probability_summary(*(np.concatenate(v) for v in buffers))
                 for label,buffers in probability_buffers.items()}
    write_json('consistency_checks.json',consistency)
    control_summary=[]
    for history,crel in itertools.product(('all','same','different'),('all','equal','unequal','missing')):
        entries=[r for r in control_rows if (history=='all' or r['history_relation']==history)
                 and (crel=='all' or r['control_relation']==crel)]
        ds=[r['control_L2'] for r in entries if r['control_L2'] is not None]
        control_summary.append({'history_relation':history,'control_relation':crel,'seed_d_direction_n':len(entries),
            'L2_mean':float(np.mean(ds)) if ds else None,'L2_median':float(np.median(ds)) if ds else None,
            'L2_min':float(min(ds)) if ds else None,'L2_max':float(max(ds)) if ds else None})
    csv_write('control_state_summary.csv',control_summary)
    summary={'status':'COMPLETED','dataset':'v1.3/B1','seeds':list(SEEDS),
        'primary':next(r for r in overall_rows if r['family']=='overall' and r['selection']=='triple_distinct'),
        'cross_all':next(r for r in overall_rows if r['family']=='overall' and r['selection']=='all'),
        'by_control':[r for r in overall_rows if r['family']=='control_relation'],
        'control_distances':control_summary,'same_start_consistency':consistency,
        'direct_index_checks':len(direct_rows),'scored_cells':len(index_rows),
        'training_updates':0,'network_forward_calls':0,'GPU_seconds_added':0,
        'torch_imported':'torch' in sys.modules,'elapsed_CPU_wall_seconds':time.perf_counter()-started,
        'precision_note':plan['precision_note']}
    write_json('summary.json',summary)
    verify()
    write_json('status.json',{'status':'COMPLETED','scored_seeds':list(SEEDS),'primary_inputs_complete':True,
                              'report_pending':True})
    print(json.dumps({'status':'COMPLETED','primary':summary['primary'],'cross_all':summary['cross_all']}),flush=True)


def verify():
    load_plan()
    before=read_json(OUT/'preservation_before.json')
    mismatches=[p for p,h in before.items() if not (REPO/p).exists() or sha(REPO/p)!=h]
    write_json('preservation_check.json',{'files':len(before),'mismatches':mismatches,'all_unchanged':not mismatches})
    assert not mismatches,mismatches
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','score','verify'))
    command=parser.parse_args().command
    {'prepare':prepare,'score':score,'verify':verify}[command]()
