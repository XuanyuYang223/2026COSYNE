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
from .task import suffix_inputs, tensor_inputs
from .train import load_model
from .truth import CONDITIONS, SUFFIXES, contents, inverse, matrix_apply, mismatched_indices

METRICS = ('loss_positive', 'new_error', 'native_correct', 'swap_correct', 'signed_loss',
           'native_ce', 'swap_ce', 'wrong_correct', 'wrong_cf_correct', 'answers_differ',
           'wrong_correct_different', 'wrong_cf_correct_different', 'tv')


@torch.no_grad()
def continue_pairs(model, memory, control, donors, receivers, suffix, cfg):
    """All q x both query types x all query IDs for each ordered pair."""
    dev = next(model.parameters()).device
    n, nq = cfg['n'], memory.shape[1]
    kind = np.repeat(np.arange(2), n)
    item = np.tile(np.arange(n), 2)
    future = suffix_inputs([suffix] * (2*n), kind, item, cfg)
    base = tensor_inputs(future, dev)
    query = torch.as_tensor(future['query'], device=dev)
    donors, receivers = np.asarray(donors), np.asarray(receivers)
    rows = len(donors) * nq * 2*n
    outputs = np.empty((rows, n), np.float32)
    batch = cfg['execution']['evaluation_batch_size']
    for start in range(0, rows, batch):
        index = np.arange(start, min(start + batch, rows))
        pair = index // (nq * 2*n)
        qindex = (index // (2*n)) % nq
        query_index = index % (2*n)
        qi = torch.as_tensor(query_index, device=dev)
        x = {key: value[qi] for key, value in base.items()}
        m = torch.as_tensor(memory[donors[pair], qindex], device=dev)
        c = torch.as_tensor(control[receivers[pair]], device=dev)
        logits = model(x, query[qi], m, c)
        outputs[start:start + len(index)] = logits.log_softmax(-1).cpu().numpy()
    return outputs.reshape(len(donors), nq, 2, n, n)


def correct_logp(logp, answers):
    return np.take_along_axis(logp, np.asarray(answers)[..., None], axis=-1)[..., 0]


def answers_from_content(q, suffix, n):
    final = np.stack([matrix_apply(q, suffix, r) for r, h in CONDITIONS])
    return final, np.stack((final, inverse(final)), axis=2)


def point_metrics(native, matched, wrong, answers, cf_answers):
    # Outputs are indexed donor,receiver,q,type,item,class; native lacks donor axis.
    base_lp = correct_logp(native, answers)
    swap_lp = correct_logp(matched, answers[None])
    nc = native.argmax(-1) == answers
    sc = matched.argmax(-1) == answers[None]
    wc = wrong.argmax(-1) == answers[None]
    wcf = wrong.argmax(-1) == cf_answers[None]
    differs = answers != cf_answers
    signed = base_lp[None] - swap_lp
    shape = matched.shape[:-1]
    broadcast = lambda x: np.broadcast_to(x[None], shape)
    tv = 0.5 * np.abs(np.exp(matched) - np.exp(native)[None]).sum(axis=-1)
    values = (np.maximum(0, signed), broadcast(nc) & ~sc, broadcast(nc), sc, signed,
              broadcast(-base_lp), -swap_lp, wc, wcf, broadcast(differs),
              wc & differs[None], wcf & differs[None], tv)
    return np.stack(values, axis=-1).astype(np.float64)


def row_metrics(points):
    means = points.mean(axis=tuple(range(points.ndim - 1)))
    result = dict(zip(METRICS, map(float, means)))
    count = result['answers_differ']
    result['wrong_correct_different'] = result['wrong_correct_different'] / count if count else float('nan')
    result['wrong_cf_correct_different'] = result['wrong_cf_correct_different'] / count if count else float('nan')
    return result


def evaluate_interventions(model, cache, cfg, out, seed, suffixes=SUFFIXES, pilot=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    q, ids = cache['content'], cache['orbit_id']
    mismatch = mismatched_indices(ids, cfg['analysis']['mismatch_seed'])
    np.savez_compressed(out / 'mismatch.npz', indices=mismatch, recipient=q, donor=q[mismatch], orbit_id=ids)
    pairs = list(itertools.combinations(range(18), 2))
    grid_donor, grid_rec = np.indices((18, 18))
    same_context = np.array([CONDITIONS[a][0] == CONDITIONS[b][0]
                             for a, b in zip(grid_donor.ravel(), grid_rec.ravel())])
    full_don = grid_donor.ravel()[same_context]
    full_rec = grid_rec.ravel()[same_context]
    labels, direction_rows, strata = [], [], []
    orbit_values, all_d, all_s = [], [], []
    unique_ids = np.unique(ids)
    sham_max, full_max = 0., 0.
    started = time.perf_counter()
    for di, d in enumerate(cfg['timing']['delays']):
        m, c = cache['memory'][di], cache['control'][di]
        for suffix in suffixes:
            native = continue_pairs(model, m, c, range(18), range(18), suffix, cfg)
            sham = continue_pairs(model, m.copy(), c.copy(), range(18), range(18), suffix, cfg)
            sham_difference = float(np.max(np.abs(native - sham)))
            assert sham_difference == 0, f'Sham outputs changed: {sham_difference}'
            matched = continue_pairs(model, m, c, grid_donor.ravel(), grid_rec.ravel(), suffix, cfg)
            matched = matched.reshape(18, 18, *native.shape[1:])
            wrong = continue_pairs(model, m[:, mismatch], c, grid_donor.ravel(), grid_rec.ravel(), suffix, cfg)
            wrong = wrong.reshape(18, 18, *native.shape[1:])
            # Entire donor state; same-context comparison must equal donor native.
            full = continue_pairs(model, m, c, full_don, full_don, suffix, cfg)
            expected = native[full_don]
            full_difference = float(np.max(np.abs(full - expected)))
            assert np.allclose(full, expected, rtol=1e-5, atol=1e-5), f'Full-state continuation failed {full_difference}'
            sham_max, full_max = max(sham_max, sham_difference), max(full_max, full_difference)
            final, answers = answers_from_content(q, suffix, cfg['n'])
            cf_final, cf_answers = answers_from_content(q[mismatch], suffix, cfg['n'])
            points = point_metrics(native, matched, wrong, answers, cf_answers)
            # Each orbit block preserves all pairs, directions and query types.
            orbit_sum = np.stack([points[:, :, ids == orb].sum(axis=(2, 4)) for orb in unique_ids], axis=2)
            orbit_values.append(orbit_sum)
            all_d.append(d)
            all_s.append(suffix)
            np.savez_compressed(out / f'raw_d{d}_S{suffix}.npz',
                                native_logp=native, sham_logp=sham, matched_logp=matched,
                                wrong_logp=wrong, full_logp=full, full_donor=full_don, full_receiver=full_rec,
                                content=q, orbit_id=ids, mismatch_index=mismatch,
                                final_content=final, answers=answers, wrong_final_content=cf_final,
                                counterfactual_answers=cf_answers, d=d, suffix=suffix, seed=seed,
                                metric_names=np.array(METRICS), per_content_metrics=points.mean(axis=4))
            for a, b in pairs:
                both = np.stack((points[a, b], points[b, a]))
                row = {'seed': seed, 'a': a, 'b': b, 'd': d, 'S': suffix,
                       'cross_context': int(CONDITIONS[a][0] != CONDITIONS[b][0]),
                       'unseen_pair': int(CONDITIONS[a][1] in ('AB', 'BA') or CONDITIONS[b][1] in ('AB', 'BA')),
                       **row_metrics(both)}
                labels.append(row)
                for donor, receiver in ((a, b), (b, a)):
                    for kind in (0, 1):
                        direction_rows.append({'seed': seed, 'donor': donor, 'receiver': receiver,
                                               'd': d, 'S': suffix, 'query_type': kind,
                                               **row_metrics(points[donor, receiver, :, kind])})
            native_correct = native.argmax(-1) == answers
            for r in (0, 1):
                idx = [a for a in range(18) if CONDITIONS[a][0] == r]
                for kind in (0, 1):
                    strata.append({'seed': seed, 'context': r, 'd': d, 'S': suffix, 'query_type': kind,
                                   'accuracy': float(native_correct[idx, :, kind].mean()),
                                   'cross_entropy': float(-correct_logp(native, answers)[idx, :, kind].mean()),
                                   'query_count': int(len(idx)*len(q)*cfg['n'])})
            print(json.dumps({'seed': seed, 'd': d, 'S': suffix,
                              'native_accuracy': float(native_correct.mean()),
                              'elapsed_seconds': time.perf_counter() - started}), flush=True)
    for filename, rows in (('labels.csv', labels), ('directions.csv', direction_rows), ('native_strata.csv', strata)):
        with (out / filename).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savez_compressed(out / 'orbit_blocks.npz', metric_sums=np.array(orbit_values),
                        orbit_ids=unique_ids, query_counts=np.array([np.sum(ids == orb)*cfg['n'] for orb in unique_ids]),
                        metric_names=np.array(METRICS), delays=np.array(all_d), suffixes=np.array(all_s),
                        seed=seed, donor_axis=np.arange(18), receiver_axis=np.arange(18), query_type=np.arange(2))
    losses, errors = np.array([r['loss_positive'] for r in labels]), np.array([r['new_error'] for r in labels])
    ceiling = bool(np.all(errors < cfg['analysis']['ceiling_new_error']) and
                   np.diff(np.quantile(losses, [0.25, 0.75]))[0] < cfg['analysis']['ceiling_loss_iqr'])
    summary = {'seed': seed, 'pilot': pilot, 'contents': len(q), 'orbits': len(unique_ids),
               'elapsed_seconds': time.perf_counter() - started, 'sham_max_logp_error': sham_max,
               'peak_gpu_mb': torch.cuda.max_memory_allocated()/2**20 if next(model.parameters()).is_cuda else 0.,
               'full_max_logp_error': full_max, 'native_min_stratum_accuracy': min(r['accuracy'] for r in strata),
               'native_strata_passed': all(r['accuracy'] >= cfg['analysis']['native_stratum_target'] for r in strata),
               'compatibility_ceiling': ceiling, 'loss_iqr': float(np.diff(np.quantile(losses, [0.25, 0.75]))[0]),
               'label_variance': float(np.var(losses)), 'mean_matched_accuracy': float(np.mean([r['swap_correct'] for r in labels])),
               'mean_wrong_accuracy': float(np.mean([r['wrong_correct'] for r in labels])),
               'mean_wrong_counterfactual_accuracy': float(np.mean([r['wrong_cf_correct'] for r in labels]))}
    write_json(out / 'intervention_summary.json', summary)
    return summary


def measure(run, cfg, manifest, pilot=False):
    seed_all(123, cfg['execution']['cpu_threads'], cfg['execution']['deterministic'])
    run = Path(run)
    model, saved = load_model(run / 'best.pt')
    if saved['seed'] not in cfg['seeds'] and not pilot:
        raise ValueError('Pilot models cannot access formal B; use --pilot for V diagnostics.')
    if next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()
    out = run / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    q, ids = contents(manifest, 'V' if pilot else 'B', behavior=not pilot)
    if pilot:
        half = len(manifest['splits']['V']['orbit_ids']) // 2
        # Fixed first whole orbits of the V behavior half, reaching 32 contents.
        wanted, count = [], 0
        for orb in manifest['splits']['V']['orbit_ids'][half:]:
            wanted.append(orb)
            count += len(manifest['orbits'][orb]['members'])
            if count >= 32:
                break
        keep = np.isin(ids, wanted)
        q, ids = q[keep], ids[keep]
    with Budget(ROOT / 'results' / 'budget.json', cfg['execution']['gpu_hour_budget']):
        cache = cache_states(model, q, ids, cfg, out / 'behavior_states.npz')
        return evaluate_interventions(model, cache, cfg, out, saved['seed'], pilot=pilot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--config')
    args = p.parse_args()
    cfg = config(args.config)
    manifest = json.loads((ROOT / 'split_manifest.json').read_text())
    measure(args.run, cfg, manifest, args.pilot)


if __name__ == '__main__':
    main()
