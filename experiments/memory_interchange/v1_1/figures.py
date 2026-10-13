from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import RESULTS, RUNS


def save(fig, name):
    out = RESULTS / 'figures'
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def build_figures():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'A': '#2563eb', 'B': '#d97706'}
    for name in ('A', 'B'):
        run = RUNS / f'candidate_{name}_seed0'
        if not (run / 'summary.json').exists():
            continue
        summary = json.loads((run / 'summary.json').read_text())
        for stage in summary['stage_summaries']:
            curve = json.loads((run / f"stage_{stage['stage']}" / 'curve.json').read_text())
            label = f'{name} / stage {stage["stage"]}'
            axes[0].plot([r['total_updates'] for r in curve], [r['V_accuracy']*100 for r in curve], '.-',
                         color=colors[name], label=label, alpha=.8)
        stages = summary['stage_summaries']
        axes[1].plot([s['stage'] for s in stages], [100*s['last_validation']['accuracy'] for s in stages], 'o-',
                     color=colors[name], label=name)
    axes[0].axhline(97, c='black', ls='--', lw=1, label='97% gate')
    axes[0].set(xlabel='Total curriculum updates', ylabel='Fixed stage V accuracy (%)', title='Different stage tasks; no full-task claim')
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(97, c='black', ls='--', lw=1)
    axes[1].set(xlabel='Curriculum stage', ylabel='Stage endpoint V accuracy (%)', xticks=range(1, 8),
                title='Both candidates stop before stage 7')
    axes[1].legend()
    save(fig, 'curriculum_learning')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ('A', 'B')):
        path = RUNS / f'candidate_{name}_seed0' / 'stage_6' / 'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        order = ['KEEP', 'I', 'C', 'II', 'CC', 'IC', 'CI']
        for offset, split, color in ((-.18, 'last_training_evaluation', '#94a3b8'), (.18, 'last_validation', colors[name])):
            strata = summary[split]['strata']['true_operation_sequence']
            values = [100*strata[k]['accuracy'] for k in order]
            bars = ax.bar(np.arange(len(order))+offset, values, .35, label='T' if offset < 0 else 'V', color=color)
            if split == 'last_validation':
                for bar, k in zip(bars, order):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(strata[k]['n']), ha='center', fontsize=7)
        ax.axhline(95, c='black', ls='--', lw=1, label='95% factor gate')
        ax.set(xticks=range(7), xticklabels=order, ylim=(0, 108), ylabel='Accuracy (%)',
               title=f'{name}: stage 6; labels above V bars are trial counts')
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
    save(fig, 'two_operation_failure')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ('A', 'B'):
        run = RUNS / f'candidate_{name}_seed0'
        if not (run / 'summary.json').exists():
            continue
        stages = json.loads((run / 'summary.json').read_text())['stage_summaries']
        values, gradients = [], []
        for stage in stages:
            diag = json.loads((run / f"stage_{stage['stage']}" / 'diagnostics.json').read_text())
            values.append(diag['V']['content_sensitivity']['query_end']['mean'])
            gradients.append(diag['actual_training_batch_gradient_path']['events']['swap_time']['gradient_l2']['mean'])
        indices = [s['stage'] for s in stages]
        axes[0].plot(indices, values, 'o-', label=name, color=colors[name])
        axes[1].semilogy(indices, np.maximum(gradients, 1e-20), 'o-', label=name, color=colors[name])
    axes[0].set(xlabel='Stage', ylabel='Final-query memory sensitivity L2 (V)', title='Sensitivity is a diagnostic, not a pass gate')
    axes[1].set(xlabel='Stage', ylabel='Loss gradient L2 at swap time (T mean)', title='Retained gradients exist; small is not None')
    for ax in axes:
        ax.set_xticks(range(1, 8))
        ax.legend()
    save(fig, 'state_path_diagnostics')
