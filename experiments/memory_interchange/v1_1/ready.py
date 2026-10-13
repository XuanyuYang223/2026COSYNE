"""Task-only final diagnostics; this module never loads G/B or fits science models."""
from __future__ import annotations

import json

import numpy as np
import torch

from ..cache import cache_states
from ..common import write_json
from ..geometry import probe_scores, ridge_fit
from ..task import trials
from ..train import load_model
from ..truth import CONDITIONS, SUFFIXES, contents, preimage
from .audit import evaluate, gate
from .common import BASE, RESULTS, Budget, model_config
from .data import load


def measure(selected, revision):
    run = BASE / selected['selected_checkpoint']
    out = RESULTS / 'task_ready'
    out.mkdir(exist_ok=True)
    model, saved = load_model(run)
    cfg = model_config(revision)
    manifest = json.loads((BASE / 'split_manifest.json').read_text())
    with Budget():
        original, raw = evaluate(model, load('original_V'), perturb=True)
        assert gate(original, revision), 'Selected stage-7 best checkpoint did not meet original behavioral factors'
        np.savez_compressed(out / 'original_V_predictions.npz', **raw)
        pool_t, ids_t = contents(manifest, 'T')
        pool_v, ids_v = contents(manifest, 'V')
        idx = np.random.default_rng(revision['probe']['seed']).choice(len(pool_t), revision['probe']['train_contents'], replace=False)
        ct = cache_states(model, pool_t[idx], ids_t[idx], cfg)
        cv = cache_states(model, pool_v, ids_v, cfg)
        x = ct['memory'].reshape(-1, 128)
        y = np.tile(np.eye(7)[ct['content']].reshape(len(idx), 49), (36, 1))
        fit = ridge_fit(x, y, revision['probe']['alpha'])
        np.savez_compressed(out / 'T_only_probe.npz', **fit, content=ct['content'], orbit_id=ct['orbit_id'])
        items, full = probe_scores(cv, fit, 7)
        # Extra v1.0 native strata use whole V contents and every query; no swaps.
        strata = []
        for r, history in CONDITIONS:
            p = preimage(pool_v, history, r)
            for d in (0, 10):
                for suffix in SUFFIXES:
                    initial = np.repeat(p, 14, axis=0)
                    kind = np.tile(np.repeat([0, 1], 7), len(p))
                    item = np.tile(np.arange(7), 2*len(p))
                    data = trials(initial, np.full(len(initial), r), np.array([history+suffix]*len(initial)),
                                  np.full(len(initial), d), kind, item, cfg)
                    metrics, raw = evaluate(model, data)
                    correct = raw['prediction'] == data['target']
                    for query_type in (0, 1):
                        mask = kind == query_type
                        strata.append({'context': r, 'H': history, 'S': suffix, 'd': d, 'query_type': query_type,
                                       'n': int(mask.sum()), 'correct': int(correct[mask].sum()),
                                       'ce_sum': float(raw['cross_entropy'][mask].sum())})
        grouped = []
        for r in (0, 1):
            for suffix in SUFFIXES:
                for d in (0, 10):
                    for query_type in (0, 1):
                        group = [s for s in strata if s['context'] == r and s['S'] == suffix and s['d'] == d and s['query_type'] == query_type]
                        count = sum(s['n'] for s in group)
                        grouped.append({'context': r, 'S': suffix, 'd': d, 'query_type': query_type, 'n': count,
                                        'accuracy': sum(s['correct'] for s in group)/count,
                                        'cross_entropy': sum(s['ce_sum'] for s in group)/count})
    result = {'candidate': selected['candidate'], 'selected_stage7_step': saved['step'], 'original_V': original,
              'probe_V_item_accuracy': items.tolist(), 'probe_V_permutation_accuracy': full.tolist(),
              'native_main_V_strata': grouped, 'native_main_V_history_counts': strata,
              'formal_G_B_opened': False, 'science_predictions_fitted': False}
    write_json(out / 'summary.json', result)
    return result
