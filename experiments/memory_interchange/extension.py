"""Unseen four-A/B words; run only after a valid main experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, Budget, config, seed_all, write_json
from .intervene import answers_from_content, continue_pairs
from .truth import CONDITIONS, TEST_WORDS, contents
from .train import load_model


def measure(run, cfg):
    run = Path(run)
    summary = json.loads((run / 'analysis' / 'intervention_summary.json').read_text())
    if not summary['native_strata_passed']:
        raise ValueError('Main native strata failed: extension cannot provide a valid interchange test.')
    seed_all(123, cfg['execution']['cpu_threads'], cfg['execution']['deterministic'])
    model, saved = load_model(run / 'best.pt')
    cached = dict(np.load(run / 'analysis' / 'behavior_states.npz'))
    out = run / 'extension'
    out.mkdir(exist_ok=True)
    rows = []
    with Budget(ROOT / 'results' / 'budget.json', cfg['execution']['gpu_hour_budget']):
        for di, d in enumerate(cfg['timing']['delays']):
            for word in TEST_WORDS:
                history, suffix = word[:2], word[2:]
                conditions = [CONDITIONS.index((r, history)) for r in (0, 1)]
                # The two naturally formed states share the same complete q.
                native = continue_pairs(model, cached['memory'][di], cached['control'][di],
                                        conditions, conditions, suffix, cfg)
                matched = continue_pairs(model, cached['memory'][di], cached['control'][di],
                                         conditions[::-1], conditions, suffix, cfg)
                final, all_answers = answers_from_content(cached['content'], suffix, cfg['n'])
                answers = all_answers[conditions]
                original_correct = native.argmax(-1) == answers
                swap_correct = matched.argmax(-1) == answers
                native_lp = np.take_along_axis(native, answers[..., None], -1)[..., 0]
                swap_lp = np.take_along_axis(matched, answers[..., None], -1)[..., 0]
                for r in (0, 1):
                    rows.append({'seed': saved['seed'], 'context': r, 'word': word, 'd': d,
                                 'new_net_transform': int(word in ('ABAB', 'BABA')),
                                 'native_accuracy': float(original_correct[r].mean()),
                                 'swap_accuracy': float(swap_correct[r].mean()),
                                 'loss_positive': float(np.maximum(0, native_lp[r]-swap_lp[r]).mean()),
                                 'new_error': float((original_correct[r] & ~swap_correct[r]).mean()),
                                 'performance_floor': int(original_correct[r].mean() < cfg['analysis']['native_stratum_target'])})
                np.savez_compressed(out / f'raw_d{d}_{word}.npz', native_logp=native, matched_logp=matched,
                                    content=cached['content'], orbit_id=cached['orbit_id'], answers=answers,
                                    contexts=np.array([0, 1]), word=word, d=d)
    with (out / 'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    measure(args.run, config())


if __name__ == '__main__':
    main()
