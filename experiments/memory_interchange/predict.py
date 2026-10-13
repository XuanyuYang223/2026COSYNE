"""Prespecified alpha=1 ridge, seed/history holdouts and block uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .common import ROOT, config, write_json
from .geometry import ridge_apply, ridge_fit
from .truth import CONDITIONS, SUFFIXES

MODELS = {'M0': (), 'M_CKA': ('cka',), 'M_raw': ('raw',), 'M_proc': ('proc',),
          'M_init': ('init_cka',), 'M_init_CKA': ('init_cka', 'cka')}
SEGMENTS = ('unseen_cross', 'unseen_same', 'seen_cross', 'seen_same')


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def merged_rows(runs):
    rows = []
    for run in runs:
        geometry = read_csv(Path(run) / 'analysis' / 'geometry.csv')
        index = {(int(g['a']), int(g['b']), int(g['d'])): g for g in geometry}
        for label in read_csv(Path(run) / 'analysis' / 'labels.csv'):
            geom = index[(int(label['a']), int(label['b']), int(label['d']))]
            result = {**geom, **label}
            rows.append({k: (v if k == 'S' else float(v)) for k, v in result.items()})
    return rows


def base_features(row):
    ra, ha = CONDITIONS[int(row['a'])]
    rb, hb = CONDITIONS[int(row['b'])]
    return [row['d'], *[float(row['S'] == s) for s in SUFFIXES], row['cross_context'],
            row['native_ce'], row['probe_item_acc'], row['control_distance'],
            sum(c != 'N' for c in ha), sum(c != 'N' for c in hb), float(ha[-1] == hb[-1]),
            float(ra == 0), float(ra == 1), float(rb == 0), float(rb == 1)]


def design(rows, name):
    return np.asarray([base_features(r) + [r[f] for f in MODELS[name]] for r in rows], np.float64)


def fit_preprocessing(x):
    """All imputation and scaling statistics are estimated from fitting rows only."""
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                       for col in x.T])
    clean = np.where(np.isfinite(x), x, median)
    mean, scale = clean.mean(axis=0), clean.std(axis=0)
    scale[scale < 1e-12] = 1.
    return {'median': median, 'mean': mean, 'scale': scale}


def transform(x, prep):
    return (np.where(np.isfinite(x), x, prep['median']) - prep['mean']) / prep['scale']


def fit_predict(x_train, y_train, x_test, alpha=1.0):
    prep = fit_preprocessing(x_train)
    fit = ridge_fit(transform(x_train, prep), y_train[:, None], alpha)
    prediction = ridge_apply(transform(x_test, prep), fit)[:, 0]
    return prediction, prep, fit


def segment(row):
    return ('unseen' if row['unseen_pair'] else 'seen') + ('_cross' if row['cross_context'] else '_same')


def context_category(row):
    return (CONDITIONS[int(row['a'])][0], CONDITIONS[int(row['b'])][0])


def errors(y, prediction):
    return {'mae': float(np.mean(np.abs(y - prediction))),
            'rmse': float(np.sqrt(np.mean((y - prediction)**2)))}


def leave_seed_out(rows, cfg):
    seeds = sorted({int(r['seed']) for r in rows})
    if seeds != sorted(cfg['seeds']):
        raise ValueError('Formal leave-one-seed-out requires all six prescribed seeds; pilot cannot be analyzed as formal data.')
    predictions, scores, coefficients = [], [], {}
    for held in seeds:
        train_idx = [i for i, r in enumerate(rows) if int(r['seed']) != held and not r['unseen_pair']]
        test_idx = [i for i, r in enumerate(rows) if int(r['seed']) == held]
        assert all(int(rows[i]['seed']) != held and not rows[i]['unseen_pair'] for i in train_idx)
        train, test = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
        ytrain = np.array([r['loss_positive'] for r in train])
        ytest = np.array([r['loss_positive'] for r in test])
        pred = {}
        for name in MODELS:
            pred[name], prep, fit = fit_predict(design(train, name), ytrain, design(test, name), cfg['analysis']['ridge_alpha'])
            coefficients[f'{held}_{name}'] = {**{k: v.tolist() for k, v in prep.items()},
                                            **{k: v.tolist() for k, v in fit.items()},
                                            'fitting_seeds': [s for s in seeds if s != held],
                                            'fitting_pairs': 91}
        for i, row in enumerate(test):
            predictions.append({'seed': held, 'a': int(row['a']), 'b': int(row['b']), 'd': int(row['d']),
                                'S': row['S'], 'segment': segment(row), 'observed': ytest[i],
                                **{name: pred[name][i] for name in MODELS}})
        for seg in SEGMENTS:
            selected = np.array([segment(r) == seg for r in test])
            for name in MODELS:
                result = errors(ytest[selected], pred[name][selected])
                base = errors(ytest[selected], pred['M0'][selected])
                scores.append({'seed': held, 'segment': seg, 'model': name, 'n': int(selected.sum()),
                               **result, 'mae_improvement': base['mae'] - result['mae'],
                               'rmse_improvement': base['rmse'] - result['rmse']})
    for seg in SEGMENTS:
        selected = [r for r in predictions if r['segment'] == seg]
        ys = np.array([r['observed'] for r in selected])
        for name in MODELS:
            result = errors(ys, np.array([r[name] for r in selected]))
            base = errors(ys, np.array([r['M0'] for r in selected]))
            scores.append({'seed': 'overall', 'segment': seg, 'model': name, 'n': len(selected), **result,
                           'mae_improvement': base['mae'] - result['mae'], 'rmse_improvement': base['rmse'] - result['rmse']})
    return predictions, scores, coefficients


def orbit_resampled_labels(rows, blocks, rng):
    ids = blocks['orbit_ids']
    weights = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
    sums = blocks['metric_sums'][..., 0]  # [d/S, donor, receiver, orbit, type]
    numerator = (sums * weights[None, None, None, :, None]).sum(axis=(-1, -2))
    denominator = 2 * np.sum(blocks['query_counts'] * weights)
    lookup = {(int(d), str(s)): i for i, (d, s) in enumerate(zip(blocks['delays'], blocks['suffixes']))}
    result = []
    for r in rows:
        a, b = int(r['a']), int(r['b'])
        index = lookup[(int(r['d']), r['S'])]
        result.append((numerator[index, a, b] + numerator[index, b, a]) / (2 * denominator))
    return np.asarray(result)


def hierarchical_bootstrap(rows, runs, cfg):
    """Resample seeds and, within each seed, whole B orbits; refit fold models."""
    seeds = cfg['seeds']
    by_seed = {s: [r for r in rows if int(r['seed']) == s] for s in seeds}
    blocks = {s: dict(np.load(Path(run) / 'analysis' / 'orbit_blocks.npz')) for s, run in zip(seeds, runs)}
    x = {s: {m: design(by_seed[s], m) for m in MODELS} for s in seeds}
    seen = {s: np.array([not r['unseen_pair'] for r in by_seed[s]]) for s in seeds}
    segments = {s: np.array([segment(r) for r in by_seed[s]]) for s in seeds}
    rng = np.random.default_rng(cfg['analysis']['bootstrap_seed'])
    samples = []
    for rep in range(cfg['analysis']['bootstrap_replicates']):
        sampled = rng.choice(seeds, len(seeds), replace=True)
        if len(set(sampled)) < 2:
            continue
        y = {s: orbit_resampled_labels(by_seed[s], blocks[s], rng) for s in set(sampled)}
        # Duplicated model seeds are kept together: all copies of held seed excluded from fitting.
        predictions = {}
        for held in set(sampled):
            fitted = [int(s) for s in sampled if s != held]
            ytrain = np.concatenate([y[s][seen[s]] for s in fitted])
            predictions[held] = {}
            for m in MODELS:
                xtrain = np.concatenate([x[s][m][seen[s]] for s in fitted])
                predictions[held][m] = fit_predict(xtrain, ytrain, x[held][m], cfg['analysis']['ridge_alpha'])[0]
        for seg in SEGMENTS:
            ys = np.concatenate([y[int(s)][segments[int(s)] == seg] for s in sampled])
            p0 = np.concatenate([predictions[int(s)]['M0'][segments[int(s)] == seg] for s in sampled])
            baseline = errors(ys, p0)
            for m in MODELS:
                pm = np.concatenate([predictions[int(s)][m][segments[int(s)] == seg] for s in sampled])
                em = errors(ys, pm)
                samples.append({'replicate': rep, 'segment': seg, 'model': m,
                                'mae_improvement': baseline['mae'] - em['mae'],
                                'rmse_improvement': baseline['rmse'] - em['rmse']})
    intervals = []
    for seg in SEGMENTS:
        for m in MODELS:
            subset = [r for r in samples if r['segment'] == seg and r['model'] == m]
            intervals.append({'segment': seg, 'model': m, 'replicates': len(subset),
                              **{metric + '_95ci': np.quantile([r[metric] for r in subset], [.025, .975]).tolist()
                                 for metric in ('mae_improvement', 'rmse_improvement')}})
    return samples, intervals


def stratified_spearman(rows, cfg):
    output = []
    for seed in cfg['seeds']:
        for d in cfg['timing']['delays']:
            for s in SUFFIXES:
                for category in ((0, 0), (0, 1), (1, 1)):
                    group = [r for r in rows if int(r['seed']) == seed and r['d'] == d and r['S'] == s and context_category(r) == category]
                    x, y = np.array([r['cka'] for r in group]), np.array([r['loss_positive'] for r in group])
                    valid = np.isfinite(x) & np.isfinite(y)
                    defined = valid.sum() >= 3 and np.var(x[valid]) > cfg['analysis']['negligible_variance'] and np.var(y[valid]) > cfg['analysis']['negligible_variance']
                    output.append({'seed': seed, 'd': d, 'S': s, 'context_category': list(category),
                                   'n': int(valid.sum()), 'spearman': float(spearmanr(x[valid], y[valid]).statistic) if defined else None})
    return output


def quantile_examples(rows, cfg):
    """Fixed within-seed/d/S/context 75th percentiles; return every qualifying example."""
    selected = []
    for seed in cfg['seeds']:
        for d in cfg['timing']['delays']:
            for s in SUFFIXES:
                for category in ((0, 0), (0, 1), (1, 1)):
                    group = [r for r in rows if r['seed'] == seed and r['d'] == d and r['S'] == s and context_category(r) == category]
                    finite = [r for r in group if np.isfinite(r['cka'])]
                    if not finite:
                        continue
                    cka_cut = np.quantile([r['cka'] for r in finite], cfg['analysis']['counterexample_quantile'])
                    error_cut = np.quantile([r['new_error'] for r in finite], cfg['analysis']['counterexample_quantile'])
                    for r in finite:
                        if r['cka'] >= cka_cut and r['new_error'] >= error_cut and r['new_error'] >= cfg['analysis']['ceiling_new_error']:
                            selected.append({**r, 'cka_cutoff': float(cka_cut), 'new_error_cutoff': float(error_cut)})
    return selected


def save_csv(path, rows):
    with Path(path).open('w') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    cfg = config(args.config)
    runs = [ROOT / 'runs' / f'seed_{s}' for s in cfg['seeds']]
    rows = merged_rows(runs)
    out = ROOT / 'results' / 'prediction'
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / 'full_distribution.csv', rows)
    predictions, scores, coefficients = leave_seed_out(rows, cfg)
    save_csv(out / 'predictions.csv', predictions)
    save_csv(out / 'scores.csv', scores)
    write_json(out / 'coefficients.json', coefficients)
    samples, intervals = hierarchical_bootstrap(rows, runs, cfg)
    save_csv(out / 'bootstrap.csv', samples)
    write_json(out / 'bootstrap_intervals.json', intervals)
    write_json(out / 'spearman.json', stratified_spearman(rows, cfg))
    save_csv(out / 'quantile_counterexamples.csv', quantile_examples(rows, cfg))


if __name__ == '__main__':
    main()
