"""V-only post-pilot checks; no configuration search or scientific H2 fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import ROOT, Budget, config, seed_all, write_json
from .task import sample_trials, tensor_inputs
from .train import load_model
from .truth import contents, matrix_apply


def diagnostics(run, cfg, manifest):
    run = Path(run)
    seed_all(123, cfg['execution']['cpu_threads'], cfg['execution']['deterministic'])
    model, saved = load_model(run / 'best.pt')
    initial, _ = load_model(run / 'initial.pt')
    dev = next(model.parameters()).device
    pool, _ = contents(manifest, 'V')
    val = sample_trials(pool, cfg['training']['validation_size'],
                        np.random.default_rng(cfg['training']['validation_seed']), cfg)
    query_item = val['query'][:, :cfg['n']].argmax(-1)
    for r in (0, 1):
        for word in np.unique(val['words']):
            idx = np.flatnonzero((val['context'] == r) & (val['words'] == word))
            independently_final = matrix_apply(val['initial'][idx], word, r)
            assert np.array_equal(independently_final, val['final'][idx])
    strata = []
    perturbation_rows = []
    sensitivity = {}
    with Budget(ROOT / 'results' / 'budget.json', cfg['execution']['gpu_hour_budget']):
        predictions, losses = [], []
        with torch.no_grad():
            for start in range(0, len(val['target']), 512):
                part = {k: v[start:start+512] for k, v in val.items()}
                x = tensor_inputs(part, dev)
                query = torch.as_tensor(part['query'], device=dev)
                original = model(x, query)
                target = torch.as_tensor(part['target'], device=dev)
                predictions.extend(original.argmax(-1).cpu().tolist())
                losses.extend(F.cross_entropy(original, target, reduction='none').cpu().tolist())
                for name, stim in [('zero', torch.zeros_like(x['stim'])),
                                   ('roll', torch.roll(x['stim'], 1, 0))]:
                    changed = model({**x, 'stim': stim}, query)
                    perturbation_rows.append({'type': name, 'n': len(part['target']),
                                              'max_logit_difference': float((changed - original).abs().max()),
                                              'mean_logit_difference': float((changed - original).abs().mean()),
                                              'prediction_changes': int((changed.argmax(-1) != original.argmax(-1)).sum())})
        predicted, ce = np.asarray(predictions), np.asarray(losses)
        k = np.array([sum(c != 'N' for c in w) for w in val['words']])
        for r in (0, 1):
            for count in range(4):
                for d in cfg['timing']['delays']:
                    for kind in (0, 1):
                        mask = (val['context'] == r) & (k == count) & (val['delay'] == d) & (val['kind'] == kind)
                        strata.append({'context': r, 'non_N': count, 'd': d, 'query_type': kind,
                                       'n': int(mask.sum()), 'accuracy': float(np.mean(predicted[mask] == val['target'][mask])),
                                       'cross_entropy': float(ce[mask].mean())})
        # A paired content perturbation isolates m sensitivity, holding c and all future inputs fixed.
        # Use d=0 trials so all rows have the same real (unpadded) timing.
        rows = np.flatnonzero(val['delay'] == 0)[:128]
        part = {k: v[rows] for k, v in val.items()}
        x = tensor_inputs(part, dev)
        query = torch.as_tensor(part['query'], device=dev)
        for name, candidate in [('initial', initial), ('selected', model)]:
            with torch.no_grad():
                c, _ = candidate.control(x['ctrl'])
                original, _ = candidate.memory(torch.cat((x['stim'], c, x['phase']), -1))
                changed, _ = candidate.memory(torch.cat((torch.zeros_like(x['stim']), c, x['phase']), -1))
                sensitivity[name] = {'mean_memory_l2_by_step': (original-changed).norm(dim=-1).mean(0).cpu().tolist(),
                                     'max_memory_abs_by_step': (original-changed).abs().amax((0, 2)).cpu().tolist()}
        x_grad = {**x, 'stim': x['stim'].clone().requires_grad_(True)}
        # cuDNN needs training-mode reserve space for backward. This model has no
        # dropout or normalization, so the transition equations are unchanged.
        model.train()
        logits = model(x_grad, query)
        loss = F.cross_entropy(logits, torch.as_tensor(part['target'], device=dev))
        model.zero_grad(set_to_none=True)
        loss.backward()
        model.eval()
        sensitivity['gradient'] = {'stimulus_gradient_l2_by_step': x_grad['stim'].grad.norm(dim=-1).mean(0).cpu().tolist(),
                                   'max_gradient_during_real_stimulus': float(x_grad['stim'].grad[:, 3:8].norm(dim=-1).mean(0).max()),
                                   'finite_parameter_gradients': all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
                                   'parameter_gradient_l2': float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))}
    difference = {}
    for component in ('control', 'memory', 'readout'):
        a = torch.cat([p.detach().ravel() for p in getattr(initial, component).parameters()])
        b = torch.cat([p.detach().ravel() for p in getattr(model, component).parameters()])
        difference[component] = float((b-a).norm())
    changes = {}
    for name in ('zero', 'roll'):
        group = [r for r in perturbation_rows if r['type'] == name]
        total = sum(r['n'] for r in group)
        changes[name] = {'max_logit_difference': max(r['max_logit_difference'] for r in group),
                         'mean_logit_difference': sum(r['n'] * r['mean_logit_difference'] for r in group) / total,
                         'prediction_change_rate': sum(r['prediction_changes'] for r in group) / total}
    result = {'scope': 'PILOT_DIAGNOSTIC_V_ONLY', 'memory_dim': model.memory_dim,
              'selected_step': saved['step'], 'validation_n': len(val['target']),
              'validation_accuracy': float(np.mean(predicted == val['target'])),
              'validation_cross_entropy': float(ce.mean()), 'chance_accuracy': 1/cfg['n'],
              'uniform_cross_entropy': float(np.log(cfg['n'])), 'strata': strata,
              'stimulus_perturbations': changes, 'content_sensitivity': sensitivity,
              'parameter_changes_l2': difference,
              'prediction_frequency': np.bincount(predicted, minlength=cfg['n']).tolist(),
              'target_frequency': np.bincount(val['target'], minlength=cfg['n']).tolist()}
    write_json(run / 'diagnostics.json', result)
    np.savez_compressed(run / 'validation_predictions.npz', prediction=predicted, target=val['target'],
                        loss=ce, context=val['context'], delay=val['delay'], word=val['words'],
                        query_type=val['kind'], query_item=query_item, initial=val['initial'], final=val['final'])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    cfg = config()
    manifest = json.loads((ROOT / 'split_manifest.json').read_text())
    result = diagnostics(args.run, cfg, manifest)
    print(json.dumps({k: result[k] for k in ['memory_dim', 'validation_accuracy', 'stimulus_perturbations', 'parameter_changes_l2']}, indent=2))


if __name__ == '__main__':
    main()
