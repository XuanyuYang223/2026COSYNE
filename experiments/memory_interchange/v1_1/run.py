from __future__ import annotations

import json
import subprocess
import sys

import torch

from ..common import sha256, write_json
from .audit import audit
from .common import BASE, ROOT, RESULTS, RUNS, Budget, config
from .data import prepare
from .ready import measure
from .train import curriculum, scratch


def verify_preserved():
    record = json.loads((RESULTS / 'preserved_v1_0.json').read_text())
    mismatches = [name for name, data in record['files'].items() if sha256(BASE / name) != data['sha256']]
    if mismatches:
        raise RuntimeError(f'v1.0 immutable files changed: {mismatches}')
    assert sha256(BASE / 'split_manifest.json') == record['split_sha256']
    return {'files_checked': len(record['files']), 'all_unchanged': True,
            'original_split_sha256': record['split_sha256'], 'original_behavior_content_count': 134}


def freeze(selected=None):
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    write_json(RESULTS / 'frozen_revision.json', {'version': '1.1', 'selected_candidate': selected,
               'freeze_type': 'approved_task_ready_plan' if selected else 'failed_execution_snapshot',
               'scope': 'task_learning_only_G_B_sealed',
               'sha256': {str(p.relative_to(BASE)): sha256(p) for p in paths},
               'dataset_manifest_sha256': sha256(RESULTS / 'datasets' / 'manifest.json'),
               'split_sha256': sha256(BASE / 'split_manifest.json')})


def execute():
    revision = config()
    preservation = verify_preserved()
    write_json(RESULTS / 'preservation_check.json', preservation)
    prepare()
    with Budget():
        checked = subprocess.run([sys.executable, '-m', 'pytest', str(BASE / 'tests'), str(ROOT / 'tests'), '-q'],
                                 capture_output=True, text=True)
    (RESULTS / 'tests.txt').write_text(checked.stdout+checked.stderr)
    if checked.returncode:
        write_json(RESULTS / 'status.json', {'status': 'BLOCKED', 'reason': 'implementation_tests_failed'})
        raise RuntimeError(checked.stdout+checked.stderr)
    if not torch.cuda.is_available():
        write_json(RESULTS / 'status.json', {'status': 'BLOCKED', 'reason': 'no_suitable_gpu'})
        return
    audited = audit()
    if audited['status'] != 'PASSED':
        write_json(RESULTS / 'status.json', {'status': 'BLOCKED', 'reason': 'training_path_audit_failed'})
        return
    toy_a = scratch('A', revision)
    candidates = ['A', 'B']
    if not toy_a['passed']:
        toy_b = scratch('B', revision)
        if not toy_b['passed']:
            write_json(RESULTS / 'status.json', {'status': 'BLOCKED', 'reason': 'both_small_fit_checks_failed',
                                               'formal_G_B_opened': False})
            freeze()
            return
        candidates = ['B']
    selected, results = None, []
    for name in candidates:
        result = curriculum(name, revision)
        results.append(result)
        if result['passed']:
            selected = result
            break  # Passing A eliminates B; no result-driven science selection.
    if selected is None:
        write_json(RESULTS / 'status.json', {'status': 'BLOCKED', 'reason': 'both_prespecified_curricula_failed',
                   'candidate_failures': [{'candidate': r['candidate'], 'stage': r['failed_stage'],
                                           'total_updates': r['total_updates']} for r in results],
                   'stage7_executed': any(any(s['stage'] == 7 for s in r['stage_summaries']) for r in results),
                   'formal_seeds_executed': [], 'formal_G_B_opened': False,
                   'H1': 'not_tested', 'H2': 'not_tested', 'H3': 'not_tested'})
        freeze()
        return
    ready = measure(selected, revision)
    freeze(selected['candidate'])
    write_json(RESULTS / 'status.json', {'status': 'TASK_READY', 'selected_candidate': selected['candidate'],
               'selected_checkpoint': selected['selected_checkpoint'], 'stage7_V_accuracy': ready['original_V']['accuracy'],
               'formal_seeds_executed': [], 'formal_G_B_opened': False,
               'scope': 'This revision authorizes task-learning only; no automatic formal science run',
               'H1': 'not_tested', 'H2': 'not_tested', 'H3': 'not_tested'})


def main():
    execute()
    from .report import build_report
    build_report()


if __name__ == '__main__':
    main()
