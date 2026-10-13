import itertools
import json

import numpy as np
import pytest
import torch

from ..cache import cache_states
from ..common import ROOT, config, seed_all
from ..model import MemoryControl
from ..task import prefix_inputs, suffix_inputs, tensor_inputs, trials
from ..truth import (CONDITIONS, HISTORIES, SUFFIXES, TRAIN_WORDS, apply, build_manifest,
                     contents, inverse, matrix_apply, mismatched_indices, net_transform_counts, orbits, preimage)


@pytest.fixture(scope='module')
def cfg():
    return config()


def test_independent_algebra_and_orbits():
    for n, expected in ((6, 115), (7, 694)):
        groups = orbits(n)
        assert len(groups) == expected
        assert {len(g) for g in groups} <= {2, 4, 8}
        ps = np.array(list(itertools.permutations(range(n))))
        assert sum(map(len, groups)) == len(ps)
        assert np.array_equal(inverse(inverse(ps)), ps)
        for r in (0, 1):
            for w in (''.join(x) for x in itertools.product('NAB', repeat=4)):
                assert np.array_equal(apply(ps, w, r), matrix_apply(ps, w, r))
            for h in HISTORIES:
                assert np.array_equal(matrix_apply(preimage(ps, h, r), h, r), ps)
        assert not np.array_equal(apply(ps, 'AB'), apply(ps, 'BA'))
    result = net_transform_counts()
    assert result == {'training_words': 65, 'unseen_words': 16, 'training_net_transforms': 7,
                      'union_net_transforms': 8, 'new_transform_words': ['ABAB', 'BABA']}


def test_split_leakage_and_mismatch(cfg):
    m = build_manifest(cfg)
    sets = [set(map(tuple, contents(m, name)[0])) for name in ('T', 'V', 'G', 'B')]
    for a, b in itertools.combinations(sets, 2):
        assert a.isdisjoint(b)
    for s in sets:
        for p in s:
            assert tuple(inverse(p)) in s and tuple(cfg['n'] - 1 - np.array(p)) in s
    q, ids = contents(m, 'B', behavior=True)
    idx = mismatched_indices(ids, cfg['analysis']['mismatch_seed'])
    assert sorted(idx.tolist()) == list(range(len(q)))
    assert np.all(ids != ids[idx])
    assert len(q) >= 128
    assert len(q) - len(m['orbits'][m['behavior_orbit_ids'][-1]]['members']) < 128
    assert len(CONDITIONS) == 18
    pairs = list(itertools.combinations(range(18), 2))
    assert len(pairs) == 153
    mixed = lambda i: CONDITIONS[i][1] in ('AB', 'BA')
    unseen = [(a, b) for a, b in pairs if mixed(a) or mixed(b)]
    assert len(unseen) == 62
    assert sum(CONDITIONS[a][0] != CONDITIONS[b][0] for a, b in unseen) == 32
    assert all(h + s in TRAIN_WORDS for h in HISTORIES for s in SUFFIXES)


def test_timing_and_final_targets(cfg):
    n = cfg['n']
    ps = np.array([np.arange(n), np.arange(n)[::-1]])
    x = trials(ps, np.array([0, 1]), np.array(['ABNA', 'BANN']), np.array([0, 10]),
               np.array([0, 1]), np.array([2, 5]), cfg)
    assert x['lengths'].tolist() == [37, 47]
    assert np.count_nonzero(x['stim'][:, :3]) == 0
    assert np.count_nonzero(x['stim'][:, 8:]) == 0
    assert np.count_nonzero(x['ctrl'][:, :34, 10:]) == 0
    for i, (r, word) in enumerate(zip(x['context'], x['words'])):
        q = matrix_apply(ps[i], word, r)
        expected = q[2] if i == 0 else np.flatnonzero(q == 5)[0]
        assert x['target'][i] == expected
    pref = prefix_inputs(ps, np.array([0, 1]), ['AB', 'BA'], np.array([0, 10]), cfg)
    assert pref['lengths'].tolist() == [21, 31]
    assert np.count_nonzero(pref['ctrl'][:, :, 10:]) == 0


def test_control_isolation_cache_sham_full_and_uninterrupted(cfg):
    seed_all(321)
    model = MemoryControl(cfg).eval()
    dev = torch.device('cpu')
    q = np.array([np.arange(cfg['n']), np.arange(cfg['n'])[::-1], [1, 3, 5, 0, 6, 2, 4]])
    cache = cache_states(model, q, np.array([0, 1, 2]), cfg)
    for di, d in enumerate(cfg['timing']['delays']):
        for a, (r, h) in enumerate(CONDITIONS):
            kind, item = np.array([0, 1, 0]), np.array([0, 4, 6])
            future = suffix_inputs(['NA'] * len(q), kind, item, cfg)
            future_t = tensor_inputs(future, dev)
            query = torch.from_numpy(future['query'])
            m = torch.from_numpy(cache['memory'][di, a])
            c = torch.from_numpy(cache['control'][di, a]).expand(len(q), -1)
            with torch.no_grad():
                original = model(future_t, query, m, c)
                sham = model(future_t, query, m.clone(), c.clone())
                torch.testing.assert_close(original, sham, rtol=0, atol=0)
                p = cache['initial'][di, a]
                full_trial = trials(p, np.full(len(q), r), np.array([h + 'NA'] * len(q)),
                                    np.full(len(q), d), kind, item, cfg)
                uninterrupted = model(tensor_inputs(full_trial, dev), query)
                torch.testing.assert_close(original, uninterrupted, atol=2e-7, rtol=2e-6)
                # Same-context full-state swap equals a donor's uninterrupted continuation.
                b = 0 if r == 0 else 9
                donor_m = torch.from_numpy(cache['memory'][di, b])
                donor_c = torch.from_numpy(cache['control'][di, b]).expand(len(q), -1)
                swapped = model(future_t, query, donor_m, donor_c)
                donor_trial = trials(cache['initial'][di, b], np.full(len(q), r),
                                     np.array([CONDITIONS[b][1] + 'NA'] * len(q)), np.full(len(q), d), kind, item, cfg)
                donor_native = model(tensor_inputs(donor_trial, dev), query)
                torch.testing.assert_close(swapped, donor_native, atol=2e-7, rtol=2e-6)
    # Vary the stimulus while preserving control input: c stays bitwise identical.
    pref = prefix_inputs(q, np.zeros(len(q), int), ['AB'] * len(q), np.zeros(len(q), int), cfg)
    with torch.no_grad():
        _, c = model.states(tensor_inputs(pref, dev))
    assert torch.equal(c[0], c[1]) and torch.equal(c[0], c[2])
