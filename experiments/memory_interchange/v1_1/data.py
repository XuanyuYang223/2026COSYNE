from __future__ import annotations

import itertools
import json

import numpy as np

from ..common import config as old_config, sha256, write_json
from ..task import sample_trials, trials
from ..truth import contents
from .common import BASE, RESULTS, config, model_config


def balanced(values, size, rng):
    values = np.asarray(values)
    whole, remainder = divmod(size, len(values))
    result = np.concatenate((np.tile(values, whole), rng.choice(values, remainder, replace=False)))
    rng.shuffle(result)
    return result


def generate(pool, size, rng, revision, stage, exact_balance=False):
    pick = (lambda values, count: balanced(values, count, rng)) if exact_balance else (lambda values, count: rng.choice(values, count))
    cfg = model_config(revision, stage)
    counts = pick(list(range(stage['max_non_N']+1)), size)
    words = np.empty(size, dtype='<U4')
    for k in range(stage['max_non_N']+1):
        candidates = [''.join(w) for w in itertools.product('NAB', repeat=4) if sum(c != 'N' for c in w) == k]
        mask = counts == k
        words[mask] = pick(candidates, int(mask.sum()))
    # Uniform T contents; none of this module can load G/B model inputs.
    p = pool[rng.integers(len(pool), size=size)]
    return trials(p, pick(stage['contexts'], size), words, pick(cfg['timing']['delays'], size),
                  pick([0, 1], size), pick(list(range(cfg['n'])), size), cfg)


def original_v():
    stored = dict(np.load(BASE / 'results' / 'validation_trials.npz'))
    item = stored['query'][:, :7].argmax(-1)
    x = trials(stored['initial'], stored['context'], stored['words'], stored['delay'], stored['kind'], item, old_config())
    for key in stored:
        assert np.array_equal(x[key], stored[key]), f'Original validation changed: {key}'
    return x


def oracle(x):
    """Recover all semantics from actual collated input; no truth/task label helpers."""
    predicted = []
    for row, length in enumerate(x['lengths']):
        stim, ctrl, phase = x['stim'][row, :length], x['ctrl'][row, :length], x['phase'][row, :length]
        assert np.all((stim == 0) | (stim == 1))
        assert np.array_equal(ctrl[:, 5:10], phase)
        assert np.all(phase.sum(-1) == 1)
        assert np.count_nonzero(x['stim'][row, length:]) == 0
        assert np.count_nonzero(x['ctrl'][row, length:]) == 0
        assert np.count_nonzero(x['phase'][row, length:]) == 0
        ctx = np.flatnonzero(phase[:, 0])
        assert len(ctx) and np.all(ctrl[ctx, :2].sum(-1) == 1)
        r = int(ctrl[ctx[0], :2].argmax())
        assert np.all(ctrl[ctx, :2].argmax(-1) == r)
        assert np.count_nonzero(ctrl[phase[:, 0] == 0, :2]) == 0
        st = np.flatnonzero(phase[:, 1])
        assert len(st)
        matrix = stim[st[0]].reshape(7, 7).astype(int)
        assert np.all(matrix.sum(0) == 1) and np.all(matrix.sum(1) == 1)
        assert np.all(stim[st] == stim[st[0]])
        assert np.count_nonzero(stim[phase[:, 1] == 0]) == 0
        rule_times = np.flatnonzero(phase[:, 3])
        assert len(rule_times) == 4
        assert np.all(ctrl[rule_times, 2:5].sum(-1) == 1)
        assert np.count_nonzero(ctrl[phase[:, 3] == 0, 2:5]) == 0
        for t in rule_times:
            cue = int(ctrl[t, 2:5].argmax())
            if cue == 0:
                continue
            if (r == 0 and cue == 1) or (r == 1 and cue == 2):
                matrix = matrix.T
            else:
                matrix = matrix[:, ::-1]
        qt = np.flatnonzero(phase[:, 4])
        assert len(qt) == (1 if len(ctx) == 1 else 3), 'Final query duration was truncated'
        assert qt[-1] == length-1 and qt[0] > rule_times[-1]
        assert np.count_nonzero(ctrl[:qt[0], 10:]) == 0
        last = ctrl[qt[-1], 10:]
        assert last[:7].sum() == 1 and last[7:].sum() == 1
        assert np.all(ctrl[qt, 10:] == last)
        assert np.array_equal(last, x['query'][row])
        item, kind = int(last[:7].argmax()), int(last[7:].argmax())
        predicted.append(int(matrix[item].argmax() if kind == 0 else matrix[:, item].argmax()))
    return np.asarray(predicted)


def factors(x):
    sequences = []
    for r, word in zip(x['context'], x['words']):
        sequences.append(''.join(('I' if (cue == 'A') == (r == 0) else 'C') for cue in word if cue != 'N') or 'KEEP')
    return {'context': np.asarray(x['context']).astype(str), 'true_operation_sequence': np.array(sequences),
            'delay': np.asarray(x['delay']).astype(str), 'query_type': np.asarray(x['kind']).astype(str),
            'non_N': np.array([sum(c != 'N' for c in w) for w in x['words']]).astype(str)}


def save_dataset(path, x):
    np.savez_compressed(path, **x)
    assert np.array_equal(oracle(x), x['target'])
    return {'path': str(path.relative_to(BASE)), 'sha256': sha256(path), 'trials': len(x['target']),
            'lengths': dict(zip(*[v.tolist() for v in np.unique(x['lengths'], return_counts=True)])),
            'oracle_accuracy': 1.0,
            'factor_counts': {name: {str(value): int(np.sum(array == value)) for value in np.unique(array)}
                              for name, array in factors(x).items()}}


def prepare():
    revision = config()
    manifest = json.loads((BASE / 'split_manifest.json').read_text())
    out = RESULTS / 'datasets'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        return json.loads((out / 'manifest.json').read_text())
    pools = {s: contents(manifest, s)[0] for s in ('T', 'V')}
    final = revision['stages'][-1]
    records = {}
    for s in ('T', 'V'):
        x = generate(pools[s], 8192, np.random.default_rng(revision['validation']['diagnostic_seed']), revision, final, True)
        records[f'diagnostic_{s}'] = save_dataset(out / f'diagnostic_{s}.npz', x)
    original = original_v()
    records['original_V'] = save_dataset(out / 'original_V.npz', original)
    for stage in revision['stages']:
        for split in ('T', 'V'):
            if stage['stage'] == 7 and split == 'V':
                x = original
            else:
                size = 8192 if stage['stage'] == 7 else 4096
                seed = revision['validation']['diagnostic_seed' if split == 'T' else 'stage_seed']
                x = generate(pools[split], size, np.random.default_rng(seed), revision, stage, True)
            key = f"stage{stage['stage']}_{split}"
            records[key] = save_dataset(out / f'{key}.npz', x)
    toy_p = np.repeat(pools['T'][:16], 14, axis=0)
    toy = trials(toy_p, np.zeros(224, int), np.array(['NNNN']*224), np.zeros(224, int),
                 np.tile(np.repeat([0, 1], 7), 16), np.tile(np.arange(7), 32), model_config(revision, revision['stages'][0]))
    records['scratch'] = save_dataset(out / 'scratch.npz', toy)
    document = {'seed_diagnostic': 20261013, 'seed_stage_validation': 20261014,
                                     'original_V_source_sha256': sha256(BASE / 'results' / 'validation_trials.npz'),
                                     'split_sha256': sha256(BASE / 'split_manifest.json'), 'datasets': records}
    write_json(out / 'manifest.json', document)
    return document


def load(name):
    return dict(np.load(RESULTS / 'datasets' / f'{name}.npz'))
