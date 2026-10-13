"""Array truth and an independent permutation-matrix evaluator."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter

import numpy as np

from .common import ROOT, config, sha256, write_json

HISTORIES = ('NN', 'NA', 'AN', 'NB', 'BN', 'AA', 'BB', 'AB', 'BA')
SUFFIXES = ('NN', 'NA', 'NB')
CONDITIONS = tuple((r, h) for r in (0, 1) for h in HISTORIES)
TRAIN_WORDS = tuple(''.join(w) for w in itertools.product('NAB', repeat=4)
                    if sum(c != 'N' for c in w) <= 3)
TEST_WORDS = tuple(''.join(w) for w in itertools.product('AB', repeat=4))


def inverse(p):
    return np.argsort(np.asarray(p), axis=-1)


def complement(p):
    p = np.asarray(p)
    return p.shape[-1] - 1 - p


def operation(cue, context):
    return 'N' if cue == 'N' else ('I' if (cue == 'A') == (context == 0) else 'C')


def apply(p, word, context=0):
    q = np.array(p, dtype=np.int64, copy=True)
    for cue in word:
        op = operation(cue, context)
        if op == 'I':
            q = inverse(q)
        elif op == 'C':
            q = complement(q)
    return q


def preimage(q, history, context):
    return apply(q, history[::-1], context)


def matrix_apply(p, word, context=0):
    """Independent verification: inversion = transpose, C = column reversal."""
    p = np.asarray(p, dtype=np.int64)
    mat = np.eye(p.shape[-1], dtype=np.int64)[p]
    for cue in word:
        if cue == 'N':
            continue
        if (context == 0 and cue == 'A') or (context == 1 and cue == 'B'):
            mat = np.swapaxes(mat, -1, -2)
        else:
            mat = mat[..., ::-1]
    return np.argmax(mat, axis=-1)


def orbits(n):
    remaining = set(itertools.permutations(range(n)))
    result = []
    while remaining:
        representative = min(remaining)
        closure = {representative}
        frontier = [representative]
        while frontier:
            p = frontier.pop()
            for q in (tuple(map(int, inverse(p))), tuple(map(int, complement(p)))):
                if q not in closure:
                    closure.add(q)
                    frontier.append(q)
        members = sorted(closure)
        remaining.difference_update(closure)
        result.append(members)
    return sorted(result)


def build_manifest(cfg):
    n = cfg['n']
    groups = orbits(n)
    expected = {6: 115, 7: 694}
    assert len(groups) == expected[n]
    order = np.random.default_rng(cfg['split_seed']).permutation(len(groups))
    # Explicit rounding: floor first three orbit counts, all remainder in B.
    counts = [int(len(groups) * f) for f in cfg['split_fractions'][:3]]
    counts.append(len(groups) - sum(counts))
    split = {}
    offset = 0
    for name, count in zip(('T', 'V', 'G', 'B'), counts):
        ids = sorted(map(int, order[offset:offset + count]))
        split[name] = {'orbit_ids': ids, 'orbit_count': count,
                       'permutation_count': sum(len(groups[i]) for i in ids)}
        offset += count
    selected, size = [], 0
    for i in split['B']['orbit_ids']:
        selected.append(i)
        size += len(groups[i])
        if size >= cfg['behavior_min_permutations']:
            break
    payload = {'n': n, 'split_seed': cfg['split_seed'], 'rounding': 'floor T/V/G; remainder B',
               'orbits': [{'id': i, 'representative': g[0], 'members': g} for i, g in enumerate(groups)],
               'orbit_size_counts': dict(Counter(map(len, groups))), 'splits': split,
               'behavior_orbit_ids': selected, 'behavior_permutation_count': size,
               'reserved_behavior_orbit_ids': [i for i in split['B']['orbit_ids'] if i not in selected]}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    payload['payload_sha256'] = hashlib.sha256(canonical).hexdigest()
    return payload


def contents(manifest, split, behavior=False):
    ids = manifest['behavior_orbit_ids'] if behavior else manifest['splits'][split]['orbit_ids']
    ps, labels = [], []
    for i in ids:
        members = manifest['orbits'][i]['members']
        ps.extend(members)
        labels.extend([i] * len(members))
    return np.array(ps, dtype=np.int64), np.array(labels, dtype=np.int64)


def mismatched_indices(orbit_ids, seed):
    """Fixed cross-orbit derangement, with exactly the same donor distribution."""
    orbit_ids = np.asarray(orbit_ids)
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        idx = rng.permutation(len(orbit_ids))
        if np.all(orbit_ids[idx] != orbit_ids):
            return idx
    raise ValueError('Could not form cross-orbit derangement; no within-orbit fallback allowed.')


def net_transform_counts(n=7):
    ps = np.array(list(itertools.permutations(range(n))))
    signatures = lambda words: {apply(ps, w, 0).tobytes() for w in words}
    trained, novel = signatures(TRAIN_WORDS), signatures(TEST_WORDS)
    newly_introduced_words = [w for w in TEST_WORDS if apply(ps, w, 0).tobytes() not in trained]
    return {'training_words': len(TRAIN_WORDS), 'unseen_words': len(TEST_WORDS),
            'training_net_transforms': len(trained), 'union_net_transforms': len(trained | novel),
            'new_transform_words': newly_introduced_words}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    cfg = config(args.config)
    manifest = build_manifest(cfg)
    write_json(ROOT / 'split_manifest.json', manifest)
    write_json(ROOT / 'split_manifest.sha256.json', {'file_sha256': sha256(ROOT / 'split_manifest.json')})
    result = {'orbit_counts': {str(n): len(orbits(n)) for n in (6, 7)},
              'transform_counts': net_transform_counts(cfg['n']), 'split_counts': manifest['splits'],
              'behavior_permutations': manifest['behavior_permutation_count']}
    write_json(ROOT / 'results' / 'mathematics.json', result)
    print(json.dumps({**result, 'split_counts': {k: {f: v[f] for f in ('orbit_count', 'permutation_count')}
                                               for k, v in manifest['splits'].items()}}, indent=2))


if __name__ == '__main__':
    main()
