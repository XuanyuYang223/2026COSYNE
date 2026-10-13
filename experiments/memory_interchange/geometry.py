from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from .cache import cache_states
from .common import ROOT, Budget, config, seed_all, write_json
from .train import load_model
from .truth import CONDITIONS, contents


def centered(x):
    x = np.asarray(x, np.float64)
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x, y):
    x, y = centered(x), centered(y)
    xx, yy = x.T @ x, y.T @ y
    denominator = np.linalg.norm(xx, 'fro') * np.linalg.norm(yy, 'fro')
    if denominator <= np.finfo(np.float64).tiny:
        return float('nan')
    return float(np.sum((x.T @ y)**2) / denominator)


def reference_cka(x, y):
    """Independent sample-Gram formula, not the feature-Gram implementation."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    k, l = x @ x.T, y @ y.T
    h = np.eye(len(x)) - np.ones((len(x), len(x))) / len(x)
    k, l = h @ k @ h, h @ l @ h
    denominator = np.linalg.norm(k) * np.linalg.norm(l)
    return float(np.sum(k * l) / denominator) if denominator > 1e-28 else float('nan')


def raw_distance(x, y, epsilon=1e-12):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float(np.mean(np.sum((x - y)**2, axis=-1)) /
                 (0.5 * np.mean(np.sum(x*x + y*y, axis=-1)) + epsilon))


def procrustes_distance(x, y):
    x, y = centered(x), centered(y)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx <= np.finfo(np.float64).tiny or ny <= np.finfo(np.float64).tiny:
        return float('nan')
    x, y = x / nx, y / ny
    # Optimal orthogonal Q is used only here; states are never transformed.
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    q = u @ vt
    return float(np.sum((x @ q - y)**2))


def ridge_fit(x, y, alpha=1.0):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    weights = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return {'x_mean': xm, 'y_mean': ym, 'weights': weights}


def ridge_apply(x, fit):
    return (x - fit['x_mean']) @ fit['weights'] + fit['y_mean']


def fit_probe(model, manifest, cfg, out):
    pool, ids = contents(manifest, 'T')
    rng = np.random.default_rng(cfg['analysis']['probe_seed'])
    index = rng.choice(len(pool), size=min(len(pool), cfg['analysis']['probe_train_per_condition_delay']), replace=False)
    cache = cache_states(model, pool[index], ids[index], cfg)
    # All 18 conditions and both delays represented equally; no G/B fitting.
    x = cache['memory'].reshape(-1, model.memory_dim)
    y = np.tile(np.eye(cfg['n'])[cache['content']].reshape(len(index), -1),
                (len(CONDITIONS) * len(cfg['timing']['delays']), 1))
    fit = ridge_fit(x, y, cfg['analysis']['probe_alpha'])
    np.savez_compressed(out, **fit, train_content=cache['content'], orbit_id=cache['orbit_id'])
    return fit


def probe_scores(cache, fit, n):
    scores = ridge_apply(cache['memory'], fit).reshape(*cache['memory'].shape[:-1], n, n)
    decoded = scores.argmax(axis=-1)
    correct = decoded == cache['content'][None, None]
    return correct.mean(axis=(-1, -2)), correct.all(axis=-1).mean(axis=-1)


def measure(model_path, initial_path, cfg, manifest, out, split='G', pilot=False):
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(123, cfg['execution']['cpu_threads'], cfg['execution']['deterministic'])
    model, saved = load_model(model_path)
    if saved['seed'] not in cfg['seeds'] and not pilot:
        raise ValueError('Pilot models cannot access formal G; use --pilot for V diagnostics.')
    if next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()
    q, ids = contents(manifest, split)
    if pilot:
        # V only; geometric half is disjoint by whole orbit from pilot behavior half.
        keep = np.isin(ids, manifest['splits']['V']['orbit_ids'][:len(manifest['splits']['V']['orbit_ids']) // 2])
        q, ids = q[keep], ids[keep]
    with Budget(ROOT / 'results' / 'budget.json', cfg['execution']['gpu_hour_budget']):
        cache = cache_states(model, q, ids, cfg, out / 'geometry_states.npz')
        fit = fit_probe(model, manifest, cfg, out / 'probe.npz')
        initial, _ = load_model(initial_path)
        init = cache_states(initial, q, ids, cfg, out / 'initial_geometry_states.npz')
    item_acc, perm_acc = probe_scores(cache, fit, cfg['n'])
    rows = []
    for di, d in enumerate(cfg['timing']['delays']):
        for a, b in itertools.combinations(range(18), 2):
            x, y = cache['memory'][di, a], cache['memory'][di, b]
            rows.append({'seed': saved['seed'], 'a': a, 'b': b, 'd': d,
                         'cka': linear_cka(x, y), 'raw': raw_distance(x, y),
                         'proc': procrustes_distance(x, y),
                         'init_cka': linear_cka(init['memory'][di, a], init['memory'][di, b]),
                         'control_distance': raw_distance(cache['control'][di, a:a+1], cache['control'][di, b:b+1]),
                         'probe_item_acc': float((item_acc[di, a] + item_acc[di, b]) / 2),
                         'probe_perm_acc': float((perm_acc[di, a] + perm_acc[di, b]) / 2)})
    with (out / 'geometry.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'seed': saved['seed'], 'split': split, 'pilot': pilot,
               'contents': len(q), 'probe_item_accuracy': item_acc.tolist(),
               'probe_permutation_accuracy': perm_acc.tolist(),
               'elapsed_seconds': time.perf_counter()-started,
               'peak_gpu_mb': torch.cuda.max_memory_allocated()/2**20 if next(model.parameters()).is_cuda else 0.,
               'geometry_pairs': len(rows), 'undefined_values': sum(not np.isfinite(row[k])
                    for row in rows for k in ('cka', 'proc', 'init_cka'))}
    write_json(out / 'geometry_summary.json', summary)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--config')
    args = p.parse_args()
    cfg = config(args.config)
    run = Path(args.run)
    manifest = json.loads((ROOT / 'split_manifest.json').read_text())
    measure(run / 'best.pt', run / 'initial.pt', cfg, manifest, run / 'analysis', 'V' if args.pilot else 'G', args.pilot)


if __name__ == '__main__':
    main()
