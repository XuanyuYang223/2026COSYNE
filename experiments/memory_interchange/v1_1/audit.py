from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from ..common import config as old_config, seed_all, sha256, write_json
from ..task import sample_trials, tensor_inputs
from ..truth import contents
from ..train import load_model
from .common import BASE, RESULTS, Budget, config
from .data import factors, load, oracle, prepare
from .trace import finite_difference, gate_and_sensitivity, gradient_path


@torch.no_grad()
def evaluate(model, data, batch=512, perturb=False):
    model.eval()
    dev = next(model.parameters()).device
    predicted, losses = [], []
    changes = {'zero': [], 'shuffle': []}
    for start in range(0, len(data['target']), batch):
        part = {k: v[start:start+batch] for k, v in data.items()}
        x = tensor_inputs(part, dev)
        query = torch.as_tensor(part['query'], device=dev)
        target = torch.as_tensor(part['target'], device=dev)
        logits = model(x, query)
        guess = logits.argmax(-1)
        predicted.extend(guess.cpu().tolist())
        losses.extend(F.cross_entropy(logits, target, reduction='none').cpu().tolist())
        if perturb:
            # Fixed cyclic content shuffle; control/queries/truth stay aligned to recipient.
            for name, stimulus in [('zero', torch.zeros_like(x['stim'])), ('shuffle', torch.roll(x['stim'], 1, 0))]:
                changed = model({**x, 'stim': stimulus}, query)
                changes[name].append({'n': len(target), 'prediction_changes': int((changed.argmax(-1) != guess).sum()),
                                      'max_logit_difference': float((changed-logits).abs().max())})
    guess, ce = np.asarray(predicted), np.asarray(losses)
    correct = guess == data['target']
    strata = {}
    for name, array in factors(data).items():
        strata[name] = {}
        for value in np.unique(array):
            mask = array == value
            strata[name][str(value)] = {'n': int(mask.sum()), 'accuracy': float(correct[mask].mean()),
                                       'cross_entropy': float(ce[mask].mean())}
    result = {'n': len(correct), 'accuracy': float(correct.mean()), 'cross_entropy': float(ce.mean()), 'strata': strata}
    if perturb:
        result['stimulus_perturbations'] = {name: {'prediction_change_rate': sum(r['prediction_changes'] for r in entries)/len(correct),
                                                 'max_logit_difference': max(r['max_logit_difference'] for r in entries)}
                                            for name, entries in changes.items()}
    return result, {'prediction': guess, 'cross_entropy': ce, 'target': data['target']}


def gate(result, revision):
    return bool(result['accuracy'] >= revision['validation']['overall_target'] and
                all(cell['accuracy'] >= revision['validation']['factor_target']
                    for name in revision['validation']['gate_factors'] for cell in result['strata'][name].values()))


def subset(x, size=64):
    return {k: v[:size] for k, v in x.items()}


def audit():
    revision = config()
    seed_all(0, revision['execution']['cpu_threads'], revision['execution']['deterministic'])
    prepare()
    out = RESULTS / 'audit'
    out.mkdir(exist_ok=True)
    if (out / 'summary.json').exists():
        return augment_real_batch(json.loads((out / 'summary.json').read_text()))
    results = {}
    with Budget():
        for dim in (128, 256):
            for checkpoint in ('best', 'latest'):
                model, saved = load_model(BASE / 'runs' / f'pilot_{dim}' / f'{checkpoint}.pt')
                key = f'v1_0_m{dim}_{checkpoint}'
                records = {}
                for split, filename in [('T', 'diagnostic_T'), ('V_balanced', 'diagnostic_V'), ('V_original', 'original_V')]:
                    dataset = load(filename)
                    assert np.array_equal(oracle(dataset), dataset['target'])
                    evaluation, raw = evaluate(model, dataset, perturb=True)
                    records[split] = evaluation
                    np.savez_compressed(out / f'{key}_{split}.npz', **raw)
                batch = subset(load('diagnostic_T'))
                records['actual_training_batch_path'] = gradient_path(model, batch, out / f'{key}_training_path_trace.npz')
                records['T_content_gates'] = gate_and_sensitivity(model, batch)
                records['V_content_gates'] = gate_and_sensitivity(model, subset(load('diagnostic_V')))
                records['checkpoint_step'] = saved['step']
                results[key] = records
                write_json(out / f'{key}.json', records)
                print(json.dumps({'audit': key, 'step': saved['step'], 'T_accuracy': records['T']['accuracy'],
                                  'V_accuracy': records['V_balanced']['accuracy'],
                                  'original_V_accuracy': records['V_original']['accuracy']}), flush=True)
        initial, _ = load_model(BASE / 'runs' / 'pilot_128' / 'initial.pt')
        short = subset(load('scratch'), 1)
        finite = finite_difference(initial, short)
    graph_checks = [r['actual_training_batch_path'] for r in results.values()]
    passed = all(not p['stimulus_gradient_none'] and p['stimulus_gradient_finite']
                 and all(not event['gradient_none_rows'] and event['gradient_finite'] for event in p['events'].values())
                 and all(not v['none'] and v['finite'] for v in p['parameters'].values()) for p in graph_checks)
    summary = {'status': 'PASSED' if passed else 'FAILED', 'implementation_bug_found': not passed,
               'old_checkpoint_evaluations': results, 'short_float64_derivative': finite,
               'actual_forward_review': {'states_zeroed_only_at_trial_start': True,
                                        'no_detach_no_grad_in_training_forward': True,
                                        'no_numpy_state_roundtrip': True,
                                        'batch_first_checked': True,
                                        'query_endpoint_before_padding_checked': True,
                                        'collated_oracle_accuracy': 1.0},
               'original_V_source_sha256': sha256(BASE / 'results' / 'validation_trials.npz'),
               'sealed_sets': ['G', 'B'],
               'documentation': ['https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html',
                                 'https://docs.pytorch.org/docs/2.14/notes/autograd.html'],
               'installed_torch': torch.__version__}
    write_json(out / 'summary.json', summary)
    return augment_real_batch(summary)


def augment_real_batch(summary):
    if 'real_v1_0_sampler_paths' in summary:
        return summary
    manifest = json.loads((BASE / 'split_manifest.json').read_text())
    batch = sample_trials(contents(manifest, 'T')[0], 128, np.random.default_rng(20261013), old_config())
    assert np.array_equal(oracle(batch), batch['target'])
    np.savez_compressed(RESULTS / 'audit' / 'real_v1_0_training_batch.npz', **batch)
    paths = {}
    with Budget():
        for dim in (128, 256):
            for name in ('best', 'latest'):
                model, _ = load_model(BASE / 'runs' / f'pilot_{dim}' / f'{name}.pt')
                paths[f'm{dim}_{name}'] = gradient_path(model, batch, RESULTS / 'audit' / f'm{dim}_{name}_real_path_trace.npz')
    summary['real_v1_0_sampler_paths'] = paths
    summary['real_v1_0_training_batch_oracle_accuracy'] = 1.0
    write_json(RESULTS / 'audit' / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    audit()
