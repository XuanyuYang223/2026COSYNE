from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import ROOT, Budget, config, device, restore_rng, rng_state, seed_all, write_json
from .model import MemoryControl
from .task import sample_trials, tensor_inputs
from .truth import contents


def slice_trials(x, start, end):
    return {k: v[start:end] for k, v in x.items()}


@torch.no_grad()
def evaluate(model, dataset, dev, batch=512):
    model.eval()
    loss, correct = 0., 0
    for start in range(0, len(dataset['target']), batch):
        x = slice_trials(dataset, start, start + batch)
        logits = model(tensor_inputs(x, dev), torch.as_tensor(x['query'], device=dev))
        target = torch.as_tensor(x['target'], device=dev)
        loss += F.cross_entropy(logits, target, reduction='sum').item()
        correct += (logits.argmax(-1) == target).sum().item()
    total = len(dataset['target'])
    return {'accuracy': correct / total, 'cross_entropy': loss / total}


def checkpoint(model, optimizer, rng, step, cfg, seed, best, seconds):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state(rng),
            'step': step, 'config': cfg, 'seed': seed, 'memory_dim': model.memory_dim,
            'best_validation': best, 'training_seconds': seconds}


def train(cfg, manifest, seed, memory_dim=None, updates=None, out=None, resume=False):
    seed_all(seed, cfg['execution']['cpu_threads'], cfg['execution']['deterministic'])
    dev = device()
    out = Path(out or ROOT / 'runs' / f'seed_{seed}')
    out.mkdir(parents=True, exist_ok=True)
    model = MemoryControl(cfg, memory_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'],
                                 weight_decay=cfg['training']['weight_decay'])
    rng = np.random.default_rng(seed + 100000)
    train_pool, _ = contents(manifest, 'T')
    val_pool, _ = contents(manifest, 'V')
    val = sample_trials(val_pool, cfg['training']['validation_size'],
                        np.random.default_rng(cfg['training']['validation_seed']), cfg)
    # Same exact validation trials for every seed and capacity; stored once.
    validation_path = ROOT / 'results' / 'validation_trials.npz'
    if cfg['training']['validation_size'] != config()['training']['validation_size'] or cfg['n'] != config()['n']:
        validation_path = out / 'validation_trials.npz'
    if not validation_path.exists():
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(validation_path, initial=val['initial'], final=val['final'],
                            words=val['words'], context=val['context'], delay=val['delay'],
                            kind=val['kind'], query=val['query'], target=val['target'])
    start_step, previous_seconds = 0, 0.
    best = {'accuracy': -1., 'cross_entropy': float('inf'), 'step': 0}
    if resume and (out / 'latest.pt').exists():
        saved = torch.load(out / 'latest.pt', map_location=dev, weights_only=False)
        assert saved['seed'] == seed and saved['memory_dim'] == model.memory_dim
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], rng)
        start_step, best = saved['step'], saved['best_validation']
        previous_seconds = saved['training_seconds']
    else:
        torch.save(checkpoint(model, optimizer, rng, 0, cfg, seed, best, 0.), out / 'initial.pt')
    maximum = updates or cfg['training']['max_updates']
    started = time.perf_counter()
    curves = out / 'training.csv'
    mode = 'a' if start_step else 'w'
    with Budget(ROOT / 'results' / 'budget.json', cfg['execution']['gpu_hour_budget']) as budget, curves.open(mode) as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'validation_accuracy', 'validation_ce',
                                              'elapsed_seconds', 'updates_per_second', 'peak_gpu_mb'])
        if not start_step:
            writer.writeheader()
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for step in range(start_step + 1, maximum + 1):
            if step % 100 == 1:
                budget.check(reserve=60)
            model.train()
            data = sample_trials(train_pool, cfg['training']['batch_size'], rng, cfg)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensor_inputs(data, dev), torch.as_tensor(data['query'], device=dev))
            loss = F.cross_entropy(logits, torch.as_tensor(data['target'], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
            optimizer.step()
            interval = cfg['training']['validation_every']
            if step % interval == 0 or step == maximum:
                result = evaluate(model, val, dev)
                elapsed = previous_seconds + time.perf_counter() - started
                row = {'step': step, 'train_loss': loss.item(), 'validation_accuracy': result['accuracy'],
                       'validation_ce': result['cross_entropy'], 'elapsed_seconds': elapsed,
                       'updates_per_second': (step - start_step) / (elapsed - previous_seconds),
                       'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2**20 if dev.type == 'cuda' else 0.}
                writer.writerow(row)
                f.flush()
                improves = (result['accuracy'] > best['accuracy'] or
                            (result['accuracy'] == best['accuracy'] and result['cross_entropy'] < best['cross_entropy']))
                if improves:
                    best = {**result, 'step': step}
                state = checkpoint(model, optimizer, rng, step, cfg, seed, best, elapsed)
                torch.save(state, out / 'latest.pt')
                if improves:
                    torch.save(state, out / 'best.pt')
                print(json.dumps({'seed': seed, 'memory_dim': model.memory_dim, **row}), flush=True)
    result = {'seed': seed, 'memory_dim': model.memory_dim, 'updates': maximum, 'device': str(dev),
              'best_validation': best, 'training_seconds': previous_seconds + time.perf_counter() - started,
              'validation_target_passed': best['accuracy'] >= cfg['training']['validation_target']}
    write_json(out / 'training_summary.json', result)
    return result


def load_model(path, dev=None):
    dev = dev or device()
    saved = torch.load(path, map_location=dev, weights_only=False)
    model = MemoryControl(saved['config'], saved['memory_dim']).to(dev)
    model.load_state_dict(saved['model'])
    model.eval()
    return model, saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--memory-dim', type=int)
    p.add_argument('--updates', type=int)
    p.add_argument('--out')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--config')
    args = p.parse_args()
    cfg = config(args.config)
    manifest = json.loads((ROOT / 'split_manifest.json').read_text())
    train(cfg, manifest, args.seed, args.memory_dim, args.updates, args.out, args.resume)


if __name__ == '__main__':
    main()
