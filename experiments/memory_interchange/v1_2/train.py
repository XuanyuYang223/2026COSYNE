from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..common import device, restore_rng, rng_state, seed_all, write_json
from ..model import MemoryControl
from ..task import sample_trials, tensor_inputs
from ..truth import contents
from ..v1_1.audit import evaluate, gate, subset
from .common import BASE, RESULTS, RUNS, Budget, config
from ..v1_1.common import model_config
from .diagnostics import diagnose
from ..v1_1.data import generate, load, oracle
from ..v1_1.trace import gate_and_sensitivity, gradient_path


def initialize(revision, candidate, seed=0):
    seed_all(seed, revision['execution']['cpu_threads'], revision['execution']['deterministic'])
    model = MemoryControl(model_config(revision)).to(device())
    if candidate == 'B':
        with torch.no_grad():
            for gru in (model.memory, model.control):
                h = gru.hidden_size
                gru.bias_ih_l0[h:2*h].fill_(revision['candidates']['B']['bias_ih_z'])
                gru.bias_hh_l0[h:2*h].fill_(revision['candidates']['B']['bias_hh_z'])
    assert all(p.requires_grad for p in model.parameters())
    return model


def optimizer_for(model, revision):
    return torch.optim.Adam(model.parameters(), lr=revision['optimizer']['learning_rate'],
                            weight_decay=revision['optimizer']['weight_decay'])


def save_checkpoint(path, model, optimizer, rng, revision, candidate, stage, step, total, best, consecutive):
    torch.save({'version': '1.2', 'revision': revision, 'candidate': candidate, 'seed': revision['pilot_seed'],
                'config': model_config(revision), 'memory_dim': model.memory_dim,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state(rng),
                'stage': stage, 'step': step, 'total_updates': total, 'best_validation': best,
                'consecutive_passes': consecutive}, path)


def restore_checkpoint(path, model, optimizer, rng):
    saved = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'], rng)
    return saved


def train_step(model, optimizer, data, revision):
    model.train()
    dev = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    query = torch.as_tensor(data['query'], device=dev)
    logits = model(tensor_inputs(data, dev), query)
    loss = F.cross_entropy(logits, torch.as_tensor(data['target'], device=dev))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), revision['optimizer']['gradient_clip'])
    optimizer.step()
    return loss


def stage_diagnostics(model, stage, revision, out, step, full=True):
    results = {}
    for split in ('T', 'V'):
        data = load(f"stage{stage['stage']}_{split}")
        # Entire fixed distribution for task/perturbation; a prespecified 64-row
        # diagnostic batch for state paths and gates. No threshold on state norms.
        metrics, raw = evaluate(model, data, revision['execution']['evaluation_batch'], perturb=full)
        small = subset(data, revision['validation']['diagnostic_batch'])
        content = gate_and_sensitivity(model, small) if full else {}
        results[split] = {**metrics, **content}
        if full:
            np.savez_compressed(out / f'predictions_step{step}_{split}.npz', **raw,
                                context=data['context'], word=data['words'], delay=data['delay'], query_type=data['kind'])
    if full:
        results['actual_training_batch_gradient_path'] = gradient_path(model, subset(load(f"stage{stage['stage']}_T")),
                                                                      out / 'training_path_trace.npz')
    return results


def curriculum(candidate, revision, seed=0, pilot=True):
    revision = dict(revision, pilot_seed=seed)
    out = RUNS / (f'candidate_{candidate}_seed0' if pilot else f'seed_{seed}')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    model = initialize(revision, candidate, seed)
    optimizer = optimizer_for(model, revision)
    rng = np.random.default_rng(100000+seed)
    initial_path = out / 'initial.pt'
    if not initial_path.exists():
        save_checkpoint(initial_path, model, optimizer, rng, revision, candidate, 0, 0, 0, {}, 0)
    else:
        restore_checkpoint(initial_path, model, optimizer, rng)
    manifest = json.loads((BASE / 'split_manifest.json').read_text())
    pool = contents(manifest, 'T')[0]
    total_updates = 0
    all_summaries = []
    for stage in revision['stages']:
        stage_id = stage['stage']
        stage_out = out / f'stage_{stage_id}'
        stage_out.mkdir(exist_ok=True)
        finished = stage_out / 'summary.json'
        if finished.exists():
            summary = json.loads(finished.read_text())
            all_summaries.append(summary)
            total_updates = summary['total_updates']
            if not summary['passed']:
                break
            saved = restore_checkpoint(stage_out / 'advance.pt', model, optimizer, rng)
            total_updates = saved['total_updates']
            continue
        start_step, consecutive = 0, 0
        best = {'accuracy': -1., 'cross_entropy': 1e99, 'step': 0}
        latest = stage_out / 'latest.pt'
        if latest.exists():
            saved = restore_checkpoint(latest, model, optimizer, rng)
            start_step, total_updates = saved['step'], saved['total_updates']
            best, consecutive = saved['best_validation'], saved['consecutive_passes']
        start = time.perf_counter()
        history_path = stage_out / 'curve.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        previous_seconds = history[-1]['elapsed_seconds'] if history else 0.
        passed = consecutive >= revision['validation']['consecutive_passes']
        step = start_step
        with Budget() as budget:
            if next(model.parameters()).is_cuda:
                torch.cuda.reset_peak_memory_stats()
            for step in range(start_step+1, (start_step if passed else stage['max_updates'])+1):
                if step % 100 == 1:
                    budget.check(60)
                # Stage 7 restores the exact v1.0 sampling distribution and timing.
                if stage_id == 7:
                    data = sample_trials(pool, revision['optimizer']['batch_size'], rng, model_config(revision, stage))
                else:
                    data = generate(pool, revision['optimizer']['batch_size'], rng, revision, stage)
                loss = train_step(model, optimizer, data, revision)
                total_updates += 1
                if step % revision['validation']['every'] == 0:
                    diagnostic = stage_diagnostics(model, stage, revision, stage_out, step, full=False)
                    metrics = diagnostic['V']
                    current_pass = gate(metrics, revision)
                    consecutive = consecutive+1 if current_pass else 0
                    passed = consecutive >= revision['validation']['consecutive_passes']
                    if metrics['accuracy'] > best['accuracy'] or (metrics['accuracy'] == best['accuracy'] and metrics['cross_entropy'] < best['cross_entropy']):
                        best = {'accuracy': metrics['accuracy'], 'cross_entropy': metrics['cross_entropy'], 'step': step}
                        save_checkpoint(stage_out / 'best.pt', model, optimizer, rng, revision, candidate,
                                        stage_id, step, total_updates, best, consecutive)
                    elapsed = previous_seconds + time.perf_counter()-start
                    row = {'step': step, 'total_updates': total_updates, 'train_loss': float(loss.detach()),
                           'T_accuracy': diagnostic['T']['accuracy'], 'T_cross_entropy': diagnostic['T']['cross_entropy'],
                           'V_accuracy': metrics['accuracy'], 'V_cross_entropy': metrics['cross_entropy'],
                           'V_strata': metrics['strata'], 'T_strata': diagnostic['T']['strata'],
                           'gate_pass': current_pass, 'consecutive_passes': consecutive,
                           'elapsed_seconds': elapsed, 'updates_per_second': (step-start_step)/(elapsed-previous_seconds),
                           'peak_gpu_mb': torch.cuda.max_memory_allocated()/2**20 if next(model.parameters()).is_cuda else 0.}
                    history.append(row)
                    write_json(history_path, history)
                    save_checkpoint(latest, model, optimizer, rng, revision, candidate, stage_id, step, total_updates, best, consecutive)
                    if passed:
                        save_checkpoint(stage_out / 'advance.pt', model, optimizer, rng, revision, candidate,
                                        stage_id, step, total_updates, best, consecutive)
                    print(json.dumps({'candidate': candidate, 'stage': stage_id,
                                      **{k: v for k, v in row.items() if k not in ('V_strata', 'T_strata')}}), flush=True)
                    if passed:
                        break
            # Diagnose and retain the actual endpoint; failures never proceed.
            diagnostic = stage_diagnostics(model, stage, revision, stage_out, step, full=True)
            write_json(stage_out / 'diagnostics.json', diagnostic)
            if stage_id in (6, 7):
                diagnose(model, revision, stage_id, stage_out / 'two_step_diagnostics', 'endpoint')
            summary = {'candidate': candidate, 'stage': stage_id, 'timing': stage['timing'], 'passed': passed,
                       'updates': step, 'total_updates': total_updates, 'best_validation': best,
                       'last_validation': {k: diagnostic['V'][k] for k in ('n', 'accuracy', 'cross_entropy', 'strata')},
                       'last_training_evaluation': {k: diagnostic['T'][k] for k in ('n', 'accuracy', 'cross_entropy', 'strata')},
                       'elapsed_seconds': previous_seconds+time.perf_counter()-start,
                       'peak_gpu_mb': torch.cuda.max_memory_allocated()/2**20 if next(model.parameters()).is_cuda else 0.}
            write_json(finished, summary)
        all_summaries.append(summary)
        if not passed:
            break
        # Restore weights, optimizer and RNG from the same exact promotion point.
        saved = restore_checkpoint(stage_out / 'advance.pt', model, optimizer, rng)
        total_updates = saved['total_updates']
    result = {'candidate': candidate, 'seed': seed, 'passed': len(all_summaries) == 7 and all(s['passed'] for s in all_summaries),
              'failed_stage': next((s['stage'] for s in all_summaries if not s['passed']), None),
              'total_updates': total_updates, 'stage_summaries': all_summaries,
              'initial_checkpoint': str(initial_path.relative_to(BASE)), 'scratch_weights_used': False}
    if result['passed']:
        selected = torch.load(out / 'stage_7' / 'best.pt', map_location='cpu', weights_only=False)
        torch.save(selected, out / 'selected_stage7.pt')
        result['selected_step_in_stage7'] = selected['step']
        result['selected_checkpoint'] = str((out / 'selected_stage7.pt').relative_to(BASE))
    write_json(out / 'summary.json', result)
    return result


if __name__ == '__main__':
    curriculum('A', config())
