from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from .common import ROOT
from .truth import preimage


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    target = ROOT / 'results' / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(target / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def task_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set(xlim=(0, 12), ylim=(0, 4.5))
    ax.axis('off')
    q = np.array([2, 0, 1, 4, 3, 6, 5])
    pa, pb = preimage(q, 'AB', 0), preimage(q, 'NN', 1)
    def box(x, y, width, height, label, color):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.1',
                                    facecolor=color, edgecolor='#475569'))
        ax.text(x+width/2, y+height/2, label, ha='center', va='center', fontsize=10)
    def arrow(a, b, color='#475569'):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='->', mutation_scale=14, color=color))
    box(.2, 2.6, 3.0, 1.0, f'Donor: r=0, H=AB\np={pa.tolist()}', '#e0f2fe')
    box(.2, .8, 3.0, 1.0, f'Recipient: r=1, H=NN\np={pb.tolist()}', '#dcfce7')
    box(4., 2.7, 2.0, .8, 'Natural state\n(m_a(q), c_a)', '#e0f2fe')
    box(4., .9, 2.0, .8, 'Natural state\n(m_b(q), c_b)', '#dcfce7')
    arrow((3.3, 3.1), (3.9, 3.1))
    arrow((3.3, 1.3), (3.9, 1.3))
    ax.text(5, 2.05, f'Identical full content q\n{q.tolist()}', ha='center', fontsize=10)
    box(7., 1.8, 1.8, 1.0, 'Intervention\n(m_a(q), c_b)', '#fef3c7')
    arrow((6.1, 3.1), (7.0, 2.6), '#0284c7')
    arrow((6.1, 1.3), (7.0, 2.0), '#16a34a')
    box(9.6, 1.8, 2., 1., 'Recipient future\nS, then query', '#f1f5f9')
    arrow((8.9, 2.3), (9.5, 2.3))
    ax.text(6, .2, 'Swap after step 21+d; query first appears at step 35+d.  Only final-query CE trains the network.',
            ha='center', fontsize=10)
    ax.text(6, 4.1, 'Preserve recipient control, clock, future input and readout', ha='center', fontsize=14)
    save(fig, '01_task')


def pilot_figures(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for run in runs:
        run = Path(run)
        rows = read_rows(run / 'training.csv')
        dim = json.loads((run / 'training_summary.json').read_text())['memory_dim']
        steps = [int(r['step']) for r in rows]
        axes[0].plot(steps, [100*float(r['validation_accuracy']) for r in rows], '.-', label=f'm={dim}')
        axes[1].plot(steps, [float(r['validation_ce']) for r in rows], '.-', label=f'm={dim}')
    axes[0].axhline(100/7, ls='--', c='gray', label='Chance: 14.29%')
    axes[0].set(xlabel='Optimizer updates', ylabel='Fixed V accuracy (%)', title='Pilot failed 97% validation gate')
    axes[1].axhline(np.log(7), ls='--', c='gray', label='Uniform output: ln(7)')
    axes[1].set(xlabel='Optimizer updates', ylabel='Validation cross-entropy (nat)', title='Both capacities: 20,000 updates')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save(fig, 'pilot_training')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, run in zip(axes, runs):
        diag = json.loads((Path(run) / 'diagnostics.json').read_text())
        for name in ('initial', 'selected'):
            values = diag['content_sensitivity'][name]['mean_memory_l2_by_step']
            ax.semilogy(np.arange(1, len(values)+1), np.maximum(values, 1e-12), label=name)
        ax.axvspan(4, 8, color='#e0f2fe', label='Stimulus')
        ax.axvline(21, c='gray', ls=':', label='Swap time (d=0)')
        ax.set(xlabel='Model step', ylabel='Memory L2: stimulus vs zero input', title=f"Pilot m={diag['memory_dim']} (V only)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save(fig, 'pilot_content_sensitivity')


def formal_figures():
    root = ROOT / 'results' / 'prediction'
    rows = read_rows(root / 'full_distribution.csv')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cross, marker, label in ((0, 'o', 'Same context'), (1, '^', 'Cross context')):
        group = [r for r in rows if int(float(r['cross_context'])) == cross and np.isfinite(float(r['cka']))]
        art = ax.scatter([float(r['cka']) for r in group], [float(r['loss_positive']) for r in group],
                         c=[float(r['native_correct']) for r in group], vmin=0, vmax=1, cmap='viridis',
                         marker=marker, s=10, alpha=.6, label=label)
    fig.colorbar(art, ax=ax, label='Native accuracy')
    ax.set(xlabel='Centered linear CKA on G', ylabel='Positive swap harm on B (nat)')
    ax.legend()
    save(fig, '02_geometry_harm')
    scores = read_rows(root / 'scores.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('mae', 'rmse')):
        for i, model in enumerate(('M0', 'M_CKA', 'M_raw', 'M_proc')):
            group = [r for r in scores if r['seed'] != 'overall' and r['segment'] == 'unseen_cross' and r['model'] == model]
            values = [float(r[metric]) for r in group]
            ax.scatter(np.full(len(values), i), values, s=25)
            ax.plot([i-.2, i+.2], [np.mean(values)]*2, c='black')
            for r, value in zip(group, values):
                ax.annotate(r['seed'], (i, value), xytext=(5, 0), textcoords='offset points', fontsize=7)
        ax.set(xticks=range(4), xticklabels=['M0', '+CKA', '+Raw', '+Proc'], ylabel=f'{metric.upper()} (nat)',
               title='Held seed + unseen cross-context pairs')
    save(fig, '03_prediction_errors')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, label in zip(axes, ('new_error', 'loss_positive', 'wrong_cf_correct_different'),
                              ('New error rate', 'Positive harm (nat)', 'Wrong-content CF accuracy (different answers)')):
        for cross in (0, 1):
            values = [np.mean([float(r[key]) for r in rows if int(float(r['d'])) == d and int(float(r['cross_context'])) == cross]) for d in (0, 10)]
            ax.plot((0, 10), values, 'o-', label='Cross' if cross else 'Same')
        ax.set(xlabel='Extra delay (steps)', ylabel=label, xticks=[0, 10])
        ax.legend()
    save(fig, '04_delay_behavior')
