from __future__ import annotations

import numpy as np
import torch

from .truth import TRAIN_WORDS, apply, inverse

CONTEXT, STIM, DELAY, RULE, QUERY = range(5)


def query_answers(q, kind, item):
    candidates = np.where(np.asarray(kind)[:, None] == 0, q, inverse(q))
    return candidates[np.arange(len(q)), item]


def input_dims(n):
    return n * n, 2 + 3 + 5 + n + 2


def blank_inputs(batch, steps, n):
    return {'stim': np.zeros((batch, steps, n * n), np.float32),
            'ctrl': np.zeros((batch, steps, 2 + 3 + 5 + n + 2), np.float32),
            'phase': np.zeros((batch, steps, 5), np.float32)}


def phase_at(x, row, start, end, stage):
    x['phase'][row, start:end, stage] = 1
    x['ctrl'][row, start:end, 5 + stage] = 1


def prefix_inputs(p, context, histories, delays, cfg):
    p, context, delays = np.asarray(p), np.asarray(context), np.asarray(delays)
    n, timing = cfg['n'], cfg['timing']
    base = timing['context'] + timing['stimulus'] + timing['initial_delay']
    slot = timing['rule_cue'] + timing['rule_blank']
    lengths = base + 2 * slot + delays
    x = blank_inputs(len(p), int(lengths.max()), n)
    for row in range(len(p)):
        t = timing['context']
        phase_at(x, row, 0, t, CONTEXT)
        x['ctrl'][row, :t, context[row]] = 1
        phase_at(x, row, t, t + timing['stimulus'], STIM)
        x['stim'][row, t:t + timing['stimulus']] = np.eye(n, dtype=np.float32)[p[row]].ravel()
        t += timing['stimulus']
        phase_at(x, row, t, t + timing['initial_delay'], DELAY)
        t += timing['initial_delay']
        for cue in histories[row]:
            phase_at(x, row, t, t + timing['rule_cue'], RULE)
            x['ctrl'][row, t:t + timing['rule_cue'], 2 + 'NAB'.index(cue)] = 1
            t += timing['rule_cue']
            phase_at(x, row, t, t + timing['rule_blank'], DELAY)
            t += timing['rule_blank']
        phase_at(x, row, t, t + delays[row], DELAY)
    x['lengths'] = lengths
    return x


def suffix_inputs(suffixes, kind, item, cfg):
    timing, n = cfg['timing'], cfg['n']
    slot = timing['rule_cue'] + timing['rule_blank']
    steps = 2 * slot + timing['final_delay'] + timing['query']
    x = blank_inputs(len(suffixes), steps, n)
    for row, word in enumerate(suffixes):
        t = 0
        for cue in word:
            phase_at(x, row, t, t + timing['rule_cue'], RULE)
            x['ctrl'][row, t:t + timing['rule_cue'], 2 + 'NAB'.index(cue)] = 1
            t += timing['rule_cue']
            phase_at(x, row, t, t + timing['rule_blank'], DELAY)
            t += timing['rule_blank']
        phase_at(x, row, t, t + timing['final_delay'], DELAY)
        t += timing['final_delay']
        phase_at(x, row, t, steps, QUERY)
        x['ctrl'][row, t:, 10 + item[row]] = 1
        x['ctrl'][row, t:, 10 + n + kind[row]] = 1
    x['lengths'] = np.full(len(suffixes), steps, np.int64)
    x['query'] = np.concatenate((np.eye(n, dtype=np.float32)[item], np.eye(2, dtype=np.float32)[kind]), axis=1)
    return x


def trials(p, context, words, delays, kind, item, cfg):
    pref = prefix_inputs(p, context, [w[:2] for w in words], delays, cfg)
    suff = suffix_inputs([w[2:] for w in words], kind, item, cfg)
    lengths = pref['lengths'] + suff['lengths']
    x = blank_inputs(len(p), int(lengths.max()), cfg['n'])
    for row, start in enumerate(pref['lengths']):
        for key in ('stim', 'ctrl', 'phase'):
            x[key][row, :start] = pref[key][row, :start]
            x[key][row, start:lengths[row]] = suff[key][row]
    q = np.empty_like(p)
    for r in (0, 1):
        for w in np.unique(words):
            mask = (context == r) & (words == w)
            if mask.any():
                q[mask] = apply(p[mask], w, r)
    x.update(lengths=lengths, query=suff['query'], target=query_answers(q, kind, item),
             kind=kind, context=context, delay=delays, words=words, initial=p, final=q)
    return x


def sample_trials(pool, size, rng, cfg):
    k = rng.integers(0, 4, size)
    words = np.empty(size, dtype='<U4')
    for count in range(4):
        candidates = [w for w in TRAIN_WORDS if sum(c != 'N' for c in w) == count]
        mask = k == count
        words[mask] = rng.choice(candidates, int(mask.sum()))
    p = pool[rng.integers(len(pool), size=size)]
    return trials(p, rng.integers(0, 2, size), words, rng.choice(cfg['timing']['delays'], size),
                  rng.integers(0, 2, size), rng.integers(0, cfg['n'], size), cfg)


def tensor_inputs(x, dev):
    return {key: torch.as_tensor(x[key], device=dev) for key in ('stim', 'ctrl', 'phase', 'lengths')}
