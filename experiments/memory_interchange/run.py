"""Finite execution plan with the mandatory pilot stop rule."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, Budget, config, sha256, write_json
from .diagnose import diagnostics
from .extension import measure as extension_measure
from .figures import formal_figures, pilot_figures, task_diagram
from .geometry import measure as geometry_measure
from .intervene import measure as intervention_measure
from .train import train
from .truth import build_manifest


def freeze(cfg, selected_dim=None, status='pilot_failure'):
    files = sorted([*ROOT.glob('*.py'), *ROOT.glob('tests/*.py'), ROOT / 'config.yaml',
                    ROOT / 'protocol_v1.txt', ROOT / 'split_manifest.json', ROOT / 'requirements.txt'])
    snapshot = {'date': '2026-10-12', 'status': status, 'selected_memory_dim': selected_dim,
                'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in files}}
    write_json(ROOT / 'results' / 'frozen_plan.json', snapshot)
    return snapshot


def environment():
    import matplotlib
    import scipy
    import yaml
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'matplotlib': matplotlib.__version__, 'pyyaml': yaml.__version__,
              'cuda_available': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        result.update(gpu=p.name, gpu_total_bytes=p.total_memory)
    write_json(ROOT / 'results' / 'environment.json', result)
    return result


def completed_or_train(cfg, manifest, dim, name, seed):
    run = ROOT / 'runs' / name
    summary = run / 'training_summary.json'
    if summary.exists():
        value = json.loads(summary.read_text())
        if value['updates'] != cfg['training']['max_updates'] or value['memory_dim'] != dim or value['seed'] != seed:
            raise ValueError('Existing run does not match frozen training plan; refusing to overwrite it.')
        checkpoint = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
        assert checkpoint['config']['training'] == cfg['training']
        assert checkpoint['config']['timing'] == cfg['timing']
        assert checkpoint['config']['n'] == cfg['n']
        return value
    return train(cfg, manifest, seed, dim, out=run, resume=True)


def run_pipeline(cfg, skip_tests=False):
    environment()
    manifest = build_manifest(cfg)
    write_json(ROOT / 'split_manifest.json', manifest)
    write_json(ROOT / 'split_manifest.sha256.json', {'file_sha256': sha256(ROOT / 'split_manifest.json')})
    if not skip_tests:
        with Budget(ROOT / 'results' / 'budget.json', cfg['execution']['gpu_hour_budget']):
            completed = subprocess.run([sys.executable, '-m', 'pytest', str(ROOT / 'tests'), '-q'],
                                       capture_output=True, text=True)
        (ROOT / 'results' / 'tests.txt').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            write_json(ROOT / 'results' / 'status.json', {'status': 'BLOCKED', 'reason': 'implementation_checks_failed'})
            raise RuntimeError(completed.stdout + completed.stderr)
    task_diagram()
    if not torch.cuda.is_available():
        smoke = copy.deepcopy(cfg)
        smoke['training']['validation_size'] = 64
        smoke['execution']['evaluation_batch_size'] = 128
        train(smoke, manifest, 0, updates=5, out=ROOT / 'runs' / 'cpu_smoke')
        write_json(ROOT / 'results' / 'status.json', {'status': 'BLOCKED', 'reason': 'no_suitable_gpu',
                                                   'formal_seeds_executed': [], 'scope': 'math_tests_and_cpu_smoke'})
        return
    pilot_names, selected = [], None
    for dim in (cfg['model']['memory_dim'], cfg['model']['fallback_memory_dim']):
        name = f'pilot_{dim}'
        pilot_names.append(name)
        result = completed_or_train(cfg, manifest, dim, name, cfg['pilot_seed'])
        run = ROOT / 'runs' / name
        if not (run / 'diagnostics.json').exists():
            diagnostics(run, cfg, manifest)
        if not (run / 'analysis' / 'intervention_summary.json').exists():
            intervention_measure(run, cfg, manifest, pilot=True)
        if not (run / 'analysis' / 'geometry_summary.json').exists():
            before = time.perf_counter()
            geometry_measure(run / 'best.pt', run / 'initial.pt', cfg, manifest, run / 'analysis', split='V', pilot=True)
            write_json(run / 'analysis' / 'geometry_timing.json', {'wall_seconds': time.perf_counter()-before})
        if result['validation_target_passed']:
            selected = dim
            break
    if selected is None:
        freeze(cfg)
        write_json(ROOT / 'results' / 'status.json', {'status': 'BLOCKED', 'reason': 'both_prespecified_pilots_failed_task_gate',
                                                   'formal_seeds_executed': [], 'formal_G_B_evaluated': False,
                                                   'pilot_runs': pilot_names, 'H1': 'inconclusive', 'H2': 'not_tested', 'H3': 'not_tested'})
        pilot_figures([ROOT / 'runs' / name for name in pilot_names])
        return
    chosen = ROOT / 'runs' / f'pilot_{selected}'
    training = json.loads((chosen / 'training_summary.json').read_text())
    behavior = json.loads((chosen / 'analysis' / 'intervention_summary.json').read_text())
    estimated = 6 * (training['training_seconds'] + behavior['elapsed_seconds'] *
                     manifest['behavior_permutation_count'] / behavior['contents'] + 120)
    budget = Budget(ROOT / 'results' / 'budget.json', cfg['execution']['gpu_hour_budget'])
    write_json(ROOT / 'results' / 'pilot_decision.json', {'selected_memory_dim': selected,
               'estimated_formal_seconds_with_geometry_margin': estimated, 'remaining_budget_seconds': budget.remaining()})
    if estimated > budget.remaining():
        write_json(ROOT / 'results' / 'status.json', {'status': 'BLOCKED', 'reason': 'projected_budget_exceeded'})
        return
    frozen_path = ROOT / 'results' / 'frozen_plan.json'
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        if previous['status'] == 'before_formal':
            for name, digest in previous['sha256'].items():
                if sha256(ROOT / name) != digest:
                    raise RuntimeError(f'Frozen source changed: {name}; no automatic formal resume.')
    freeze(cfg, selected, 'before_formal')
    native_ok = True
    for seed in cfg['seeds']:
        name = f'seed_{seed}'
        result = completed_or_train(cfg, manifest, selected, name, seed)
        run = ROOT / 'runs' / name
        if not (run / 'analysis' / 'intervention_summary.json').exists():
            intervention_measure(run, cfg, manifest)
        native = json.loads((run / 'analysis' / 'intervention_summary.json').read_text())
        native_ok &= native['native_strata_passed'] and result['validation_target_passed']
        if not (run / 'analysis' / 'geometry_summary.json').exists():
            geometry_measure(run / 'best.pt', run / 'initial.pt', cfg, manifest, run / 'analysis')
    subprocess.run([sys.executable, '-m', 'experiments.memory_interchange.predict'], check=True)
    formal_figures()
    if native_ok:
        for seed in cfg['seeds']:
            run = ROOT / 'runs' / f'seed_{seed}'
            if not (run / 'extension' / 'summary.csv').exists():
                extension_measure(run, cfg)
    write_json(ROOT / 'results' / 'status.json', {'status': 'COMPLETED' if native_ok else 'EXPLORATORY',
                'reason': 'all_main_analyses_completed' if native_ok else 'some_native_task_gates_failed',
                'formal_seeds_executed': cfg['seeds'], 'formal_G_B_evaluated': True, 'pilot_runs': pilot_names,
                'extension_executed': bool(native_ok)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    args = p.parse_args()
    run_pipeline(config(args.config))
    from .report import build_report
    build_report()


if __name__ == '__main__':
    main()
