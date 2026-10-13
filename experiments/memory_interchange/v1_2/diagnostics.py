"""Fixed T/V behavioral diagnosis. No task loss changes and no G/B access."""
import csv
import itertools
import numpy as np
from ..common import write_json
from ..task import trials
from ..v1_1.audit import evaluate, subset
from ..v1_1.common import model_config
from ..v1_1.data import load, oracle
from ..v1_1.trace import gate_and_sensitivity, gradient_path

CANDIDATES = ('correct', 'first_only', 'second_only', 'none', 'reversed')

def candidate_maps(initial, operations):
    """Independent permutation-matrix oracle, including every candidate collision."""
    n = len(initial)
    matrix = np.eye(n, dtype=np.int8)[initial]
    def apply(sequence):
        state = matrix.copy()
        for op in sequence:
            state = state.T.copy() if op == 'I' else state[:, ::-1].copy()
        return state.argmax(1), state.argmax(0)
    sequences = (operations, operations[:1], operations[1:], '', operations[::-1])
    pairs = [apply(s) for s in sequences]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])

def metadata(data):
    visible = np.array([''.join(c for c in w if c != 'N') for w in data['words']])
    operations = np.array([''.join(('I' if (c == 'A') == (r == 0) else 'C')
                                  for c in w if c != 'N') for w, r in zip(data['words'], data['context'])])
    slots = np.array([','.join(str(j+1) for j,c in enumerate(w) if c != 'N') for w in data['words']])
    return {'real_operation': operations, 'context': data['context'].astype(str),
            'visible_order': visible, 'slots': slots, 'delay': data['delay'].astype(str),
            'query_type': data['kind'].astype(str)}

def table(meta, raw):
    correct = raw['prediction'] == raw['target']
    rows = []
    for name, values in meta.items():
        for value in np.unique(values):
            mask = values == value
            rows.append({'group': name, 'value': str(value), 'n': int(mask.sum()),
                         'accuracy': float(correct[mask].mean()), 'CE': float(raw['cross_entropy'][mask].mean())})
    # Only observed sparse crossed cells; descriptive, never additional gates.
    names = list(meta)
    keys = np.array(['|'.join(str(meta[n][i]) for n in names) for i in range(len(correct))])
    for key in np.unique(keys):
        mask = keys == key
        rows.append({'group': 'cross:'+'|'.join(names), 'value': key, 'n': int(mask.sum()),
                     'accuracy': float(correct[mask].mean()), 'CE': float(raw['cross_entropy'][mask].mean())})
    return rows

def save_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def rates(prediction, candidates):
    """Two separate distinguishability definitions, fixed before diagnosis."""
    out = {}
    for j, name in enumerate(CANDIDATES):
        match = prediction == candidates[:, j]
        other = np.delete(candidates, j, axis=1)
        masks = {'all': np.ones(len(match),bool),
                 'differs_from_correct': candidates[:,j] != candidates[:,0],
                 'unique_among_candidates': (other != candidates[:,j,None]).all(1)}
        out[name] = {k:{'n':int(mask.sum()), 'matches':int(match[mask].sum()),
                        'rate':float(match[mask].mean()) if mask.any() else None}
                     for k,mask in masks.items()}
    return out

def diagnose(model, revision, stage_id, out, label):
    out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    cfg = model_config(revision, revision['stages'][stage_id-1])
    for split in ('T','V'):
        full = load(f'stage{stage_id}_{split}')
        metrics, full_raw = evaluate(model, full, revision['execution']['evaluation_batch'], perturb=True)
        assert np.array_equal(oracle(full), full['target'])
        meta_full = metadata(full)
        idx = np.flatnonzero(np.array([len(s)==2 for s in meta_full['real_operation']]))
        data = {k:v[idx] for k,v in full.items()}
        meta = metadata(data)
        raw = {k:v[idx] for k,v in full_raw.items()}
        mappings, inverses = zip(*(candidate_maps(p,o) for p,o in zip(data['initial'],meta['real_operation'])))
        mappings, inverses = np.stack(mappings), np.stack(inverses)
        n = revision['n']; item = full['query'][idx,:n].argmax(1)
        answers = np.where(data['kind'][:,None]==0,
                           np.take_along_axis(mappings, item[:,None,None],axis=2).squeeze(2),
                           np.take_along_axis(inverses, item[:,None,None],axis=2).squeeze(2))
        assert np.array_equal(answers[:,0],data['target'])
        expanded = trials(np.repeat(data['initial'],2*n,axis=0), np.repeat(data['context'],2*n),
                          np.repeat(data['words'],2*n), np.repeat(data['delay'],2*n),
                          np.tile(np.repeat([0,1],n),len(idx)), np.tile(np.arange(n),2*len(idx)),cfg)
        expanded_metrics, expanded_raw = evaluate(model, expanded, revision['execution']['evaluation_batch'])
        prediction = expanded_raw['prediction'].reshape(-1,2,n)
        forward, backward = prediction[:,0], prediction[:,1]
        bij_f = (np.sort(forward,axis=1)==np.arange(n)).all(1)
        bij_b = (np.sort(backward,axis=1)==np.arange(n)).all(1)
        inverse_consistent = ((np.take_along_axis(forward,backward,axis=1)==np.arange(n)).all(1) &
                              (np.take_along_axis(backward,forward,axis=1)==np.arange(n)).all(1))
        match_f = (forward[:,None,:]==mappings).all(2)
        match_b = (backward[:,None,:]==inverses).all(2)
        joint_match = match_f & match_b
        equal_answer = answers[:,:,None] == answers[:,None,:]
        equal_mapping = (mappings[:,:,None,:]==mappings[:,None,:,:]).all(3)
        rows = table(meta,raw); save_csv(out/f'{label}_{split}_two_step_strata.csv',rows)
        map_rows = []
        for group in ('all','II','CC','IC','CI'):
            mask = np.ones(len(idx),bool) if group=='all' else meta['real_operation']==group
            if not mask.any(): continue
            row={'real_operation':group,'n':int(mask.sum()),'forward_correct':float(match_f[mask,0].mean()),
                 'inverse_correct':float(match_b[mask,0].mean()),'both_correct':float(joint_match[mask,0].mean()),
                 'forward_bijection':float(bij_f[mask].mean()),'inverse_bijection':float(bij_b[mask].mean()),
                 'mutually_inverse':float(inverse_consistent[mask].mean())}
            for j,name in enumerate(CANDIDATES):
                row['both_match_'+name]=float(joint_match[mask,j].mean())
                distinguishable = mask & ~equal_mapping[:,j,0]
                row[name+'_distinguishable_n']=int(distinguishable.sum())
                row[name+'_distinguishable_match']=float(joint_match[distinguishable,j].mean()) if distinguishable.any() else None
            map_rows.append(row)
        save_csv(out/f'{label}_{split}_mapping_summary.csv',map_rows)
        np.savez_compressed(out/f'{label}_{split}_raw.npz', dataset_row=idx, **{**data, **raw},
                            real_operation=meta['real_operation'],visible_order=meta['visible_order'],slots=meta['slots'],
                            candidate_names=np.array(CANDIDATES),candidate_answers=answers,
                            candidate_mappings=mappings,candidate_inverses=inverses,
                            candidate_answer_equal=equal_answer,candidate_mapping_equal=equal_mapping,
                            predicted_forward=forward,predicted_inverse=backward,
                            forward_bijection=bij_f,inverse_bijection=bij_b,mutually_inverse=inverse_consistent,
                            forward_candidate_match=match_f,inverse_candidate_match=match_b)
        behavior = {'all_operations':rates(raw['prediction'],answers)}
        for operation in ('II','CC','IC','CI'):
            mask = meta['real_operation']==operation
            behavior[operation]=rates(raw['prediction'][mask],answers[mask])
        learning = {'fixed_first_64':gate_and_sensitivity(model,subset(full))}
        failures = [op for op,c in metrics['strata']['true_operation_sequence'].items()
                    if c['accuracy'] < revision['validation']['factor_target']]
        for op in failures:
            select = np.flatnonzero(meta_full['real_operation']==('' if op=='KEEP' else op))[:revision['validation']['diagnostic_batch']]
            if not len(select): continue
            small = {k:v[select] for k,v in full.items()}
            learning[op]={'dataset_rows':select.tolist(),'content_gates':gate_and_sensitivity(model,small),
                          'gradient':gradient_path(model,small,out/f'{label}_{split}_{op}_gradient.npz')}
        all_results[split]={'full_task':metrics,'two_step_n':len(idx),'candidate_agreement':behavior,
                            'full_mapping':map_rows,'expanded_query_metrics':expanded_metrics,
                            'learning':learning,'failed_operation_layers':failures}
    write_json(out/f'{label}_summary.json',all_results)
    return all_results
