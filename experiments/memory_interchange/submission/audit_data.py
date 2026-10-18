"""Read saved CSVs only; validate denominators and export publication tables.

No project model modules, torch, caches, predictions, or fits are executed.
All paths in the manifest are relative to the repository root.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent
OUT = HERE / 'data'
SEEDS = range(101, 107)
SEEN = {}


def read(path):
    raw = path.read_bytes()
    SEEN[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return list(csv.DictReader(raw.decode().splitlines()))


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def close(a, b, tol=1e-12):
    assert abs(float(a) - float(b)) <= tol, (a, b)


def run():
    OUT.mkdir(exist_ok=True)
    full = read(BASE / 'results/v1_3/prediction/full_distribution.csv')
    trans_summary = read(BASE / 'Final_audit/translation_seed_summary.csv')
    native_summary = read(BASE / 'Final_audit/native_seed_summary.csv')
    donor_summary = read(BASE / 'donor_consistency/seed_summary.csv')
    donor_text = [r for r in donor_summary if r['treatment'] == 'original'
                  and all(r[k] == 'all' for k in ('context', 'history', 'S', 'd', 'query_type'))
                  and r['donor_subset'] in ('all', 'error')]
    assert len(donor_text) == 28  # 2 datasets x 2 subsets x (6 seeds + mean)
    write('donor_text_support.csv', [{k: r[k] for k in (
        'dataset', 'seed', 'donor_subset', 'n_subgroup_queries', 'answer_agree_n',
        'answer_agreement_equal_cells', 'answer_agreement_pooled')} for r in donor_text])
    geometry = read(BASE / 'Final_audit/geometry_decomposition.csv')
    geom_summary = read(BASE / 'Final_audit/geometry_distributions.csv')
    b_rows, c_rows, d_rows, d_stats = [], [], [], []
    for seed in SEEDS:
        labels = read(BASE / f'runs/v1_3/seed_{seed}/analysis/labels.csv')
        directions = read(BASE / f'runs/v1_3/seed_{seed}/analysis/directions.csv')
        old_geom = read(BASE / f'runs/v1_3/seed_{seed}/analysis/geometry.csv')
        gs_path = BASE / f'runs/v1_3/seed_{seed}/analysis/geometry_summary.json'
        SEEN[str(gs_path.relative_to(ROOT))] = hashlib.sha256(gs_path.read_bytes()).hexdigest()
        assert json.loads(gs_path.read_text())['contents'] == 492
        lookup = {(int(r['a']), int(r['b']), int(r['d'])): r for r in old_geom}
        assert len(lookup) == len(old_geom) == 306
        for cross, n_pairs in ((0, 72), (1, 81)):
            selected = [r for r in labels if int(r['cross_context']) == cross]
            assert len(selected) == n_pairs * 2 * 3
            assert len({(r['a'], r['b'], r['d'], r['S']) for r in selected}) == len(selected)
            assert {int(r['n_all']) for r in selected} == {2 * 134 * 14}
            assert {r['d'] for r in selected} == {'0', '10'}
            assert {r['S'] for r in selected} == {'NN', 'NA', 'NB'}
            rate = st.mean(float(r['matched_R_all']) for r in selected)
            error_n = sum(round(float(r['matched_R_all']) * int(r['n_all'])) for r in selected)
            close(rate, error_n / sum(int(r['n_all']) for r in selected))
            for r in selected:
                saved = next(v for v in full if all(float(v[k]) == float(r[k])
                             for k in ('seed', 'a', 'b', 'd')) and v['S'] == r['S'])
                close(saved['matched_R_all'], r['matched_R_all'])
                close(saved['n_all'], r['n_all'])
            direct = [r for r in directions if r['mask'] == 'all'
                      and int(r['r_donor'] != r['r_receiver']) == cross]
            assert len(direct) == n_pairs * 2 * 2 * 3 * 2
            assert {int(r['n']) for r in direct} == {134 * 7}
            close(rate, st.mean(float(r['matched_R']) for r in direct))
            b_rows.append(dict(seed=seed, cross=cross, pair_d_suffix_cells=len(selected),
                               query_n=sum(int(r['n_all']) for r in selected),
                               new_error_n=error_n,
                               new_error_rate=rate, new_error_percent=100 * rate))
            # The B1 Final-audit re-summary is a separate direction-level check.
            saved = next(r for r in trans_summary if r['cohort'] == 'B1'
                         and r['seed'] == str(seed) and r['cross'] == str(cross)
                         and r['mode'] == 'all' and r['S'] == r['d'] == 'ALL')
            close(rate, saved['original_new_error_original'])
            assert int(saved['query_n']) == b_rows[-1]['query_n']
            # Plot d=0 ONLY. No S axis and no directed duplication.
            cells = [r for r in geometry if int(r['seed']) == seed
                     and int(r['cross']) == cross and int(r['d']) == 0]
            assert len(cells) == n_pairs
            assert len({(r['seed'], r['a'], r['b'], r['d']) for r in cells}) == n_pairs
            for r in cells:
                a, b = int(r['a']), int(r['b'])
                assert a < b and int((a // 9) != (b // 9)) == cross
                close(r['cka'], lookup[(a, b, 0)]['cka'])
                d_rows.append({k: r[k] for k in ('seed', 'a', 'b', 'd', 'cross', 'cka')})
            median = st.median(float(r['cka']) for r in cells)
            saved = next(r for r in geom_summary if r['seed'] == str(seed)
                         and r['cross'] == str(cross) and r['d'] == '0' and r['metric'] == 'cka')
            close(median, saved['median'])
            d_stats.append(dict(seed=seed, cross=cross, d=0, n=len(cells), median=median))
        translated, native = [], []
        for d in (0, 10):
            for suffix in ('NN', 'NA', 'NB'):
                folder = BASE / f'Final_audit/B_reserved/seed_{seed}'
                tr = read(folder / f'translation_d{d}_S{suffix}.csv')
                tr = [r for r in tr if r['mode'] == 'all' and r['cross'] == '1'
                      and r['query_type'] == '-1']
                assert len(tr) == 162
                assert {int(r['n']) for r in tr} == {362 * 14}
                assert all(r['donor'] != r['receiver'] for r in tr)
                if suffix == 'NN':
                    for r in tr:
                        close(r['original_accuracy_original'], r['translated_accuracy_original'])
                translated.extend(tr)
                nr = read(folder / f'native_d{d}_S{suffix}.csv')
                nr = [r for r in nr if r['query_type'] == '-1'
                      and r['treatment'] in ('native', 'native_translated')]
                assert len(nr) == 36
                assert {int(r['n']) for r in nr} == {362 * 14}
                native.extend(nr)
        assert len(translated) == 972
        saved = next(r for r in trans_summary if r['cohort'] == 'B_reserved'
                     and r['seed'] == str(seed) and r['cross'] == '1'
                     and r['mode'] == 'all' and r['S'] == r['d'] == 'ALL')
        for cue, field in (('original', 'original_accuracy_original'),
                           ('translated', 'translated_accuracy_original')):
            val = st.mean(float(r[field]) for r in translated)
            close(val, saved[field])
            assert int(saved['query_n']) == sum(int(r['n']) for r in translated)
            c_rows.append(dict(seed=seed, treatment='Cross-context exchange', cues=cue,
                               condition_d_suffix_cells=len(translated),
                               query_n=sum(int(r['n']) for r in translated),
                               accuracy=val, accuracy_percent=100 * val))
        for cue, treatment in (('original', 'native'), ('translated', 'native_translated')):
            cells = [r for r in native if r['treatment'] == treatment]
            assert len(cells) == 108
            val = st.mean(float(r['original_target_accuracy']) for r in cells)
            saved = next(r for r in native_summary if r['cohort'] == 'B_reserved'
                         and r['seed'] == str(seed) and r['treatment'] == treatment)
            close(val, saved['original_target_accuracy'])
            c_rows.append(dict(seed=seed, treatment='No exchange', cues=cue,
                               condition_d_suffix_cells=len(cells),
                               query_n=sum(int(r['n']) for r in cells),
                               accuracy=val, accuracy_percent=100 * val))
    for cross in (0, 1):
        vals = [float(r['cka']) for r in d_rows if int(r['cross']) == cross]
        median = st.median(vals)
        saved = next(r for r in geom_summary if r['seed'] == 'all'
                     and r['cross'] == str(cross) and r['d'] == '0' and r['metric'] == 'cka')
        close(median, saved['median'])
        d_stats.append(dict(seed='all', cross=cross, d=0, n=len(vals), median=median))
    write('panel_B_seed.csv', b_rows)
    write('panel_C_seed.csv', c_rows)
    write('panel_D_d0.csv', d_rows)
    write('panel_D_medians.csv', d_stats)
    means = {f'B_{c}': st.mean(r['new_error_percent'] for r in b_rows if r['cross'] == c)
             for c in (0, 1)}
    means.update({f'C_{t}_{cue}': st.mean(r['accuracy_percent'] for r in c_rows
                                       if r['treatment'] == t and r['cues'] == cue)
                  for t in ('Cross-context exchange', 'No exchange')
                  for cue in ('original', 'translated')})
    means.update({f'D_{r["cross"]}_median': r['median'] for r in d_stats if r['seed'] == 'all'})
    (OUT / 'actual_values.json').write_text(json.dumps(means, indent=2) + '\n')
    for name in ['Final_audit/report.md', 'Final_audit/README.md',
                 'Final_audit/frozen_plan.json', 'Final_audit/B_reserved_manifest.json',
                 'Final_audit/fixed_prediction_summary.json',
                 'results/v1_3/report.md', 'results/v1_3/prediction/summary.json',
                 'results/v1_3/prediction/bootstrap_intervals.json',
                 'donor_consistency/report.md', 'donor_consistency/overall.csv',
                 'donor_consistency/control_state_differences.csv',
                 'wrong_content_recombination/report.md',
                 'v1_3/intervene.py', 'final_audit/behavior.py',
                 'final_audit/geometry_probe.py', 'final_audit/summarize.py', 'truth.py']:
        path = BASE / name
        if path.exists():
            SEEN[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = OUT / 'source_manifest.json'
    # Never silently accept a changed source on subsequent reproductions.
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())['sha256']
        assert all(SEEN.get(p) == sha for p, sha in previous.items()), 'Saved source changed'
    manifest_path.write_text(json.dumps(dict(sha256=SEEN, seed_count=6,
        model_forward_calls=0, training_updates=0, predictor_refits=0,
        B='all queries; pair/d/S equal, both directions and query types pooled, seed equal',
        C='B_reserved; all queries, d=0/10 and S=NN/NA/NB equal; true native control',
        D='G; d=0 only; one row per seed/unordered condition pair; no S replication'), indent=2) + '\n')
    print(json.dumps(means, indent=2))
    print(f'Validated {len(SEEN)} unchanged source files; no model execution.')


if __name__ == '__main__':
    run()
