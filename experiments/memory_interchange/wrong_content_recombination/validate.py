#!/usr/bin/env python3
"""Independently validate saved counts, equal-seed means and delivery hashes."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

OUT=Path(__file__).resolve().parent


def records(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def main():
    plan=json.loads((OUT/'analysis_plan.json').read_text())
    receipt=json.loads((OUT/'freeze_receipt.json').read_text())
    assert hashlib.sha256((OUT/'analysis_plan.json').read_bytes()).hexdigest()==receipt['analysis_plan_sha256']
    assert hashlib.sha256((OUT/'analysis.py').read_bytes()).hexdigest()==plan['code_sha256']['analysis.py']
    assert len(set(map(tuple,plan['content'])))==134
    assert len({(r['r'],r['H']) for r in plan['conditions']})==18
    seed=records('seed_summary.csv');strata=records('strata_summary.csv');overall=records('overall.csv')
    categories=['donor_math','receiver_original_math','recombined_math','other']
    native_cross=['donor_correct_reproduced','donor_correct_not_reproduced',
                  'donor_error_reproduced','donor_error_not_reproduced']
    lookup=defaultdict(list)
    for row in seed+strata:
        n=int(row['selected_n'])
        if row['selection']=='triple_distinct':
            assert sum(int(row[key+'_n']) for key in categories)==n
        assert sum(int(row[key+'_n']) for key in native_cross)==n
        for key in categories+['reproduce_donor_native']:
            if n:
                assert math.isclose(float(row[key+'_rate']),int(row[key+'_n'])/n,rel_tol=0,abs_tol=1e-15)
            else:
                assert row[key+'_rate']=='null'
        lookup[row['family'],row['stratum'],row['selection']].append(row)
    for row in overall:
        source=lookup[row['family'],row['stratum'],row['selection']]
        assert {int(r['seed']) for r in source}==set(range(101,107))
        assert sum(int(r['selected_n']) for r in source)==int(row['selected_n'])
        for key in categories+['reproduce_donor_native']:
            observed=[float(r[key+'_rate']) for r in source if r[key+'_rate']!='null']
            value=row['six_seed_equal_mean_'+key+'_rate']
            if len(observed)==6:
                assert math.isclose(float(value),sum(observed)/6,rel_tol=0,abs_tol=1e-15)
            else:
                assert value=='null'
        if row['family']=='S' and row['stratum']=='NN' and row['selection']=='triple_distinct':
            assert int(row['selected_n'])==0 and row['donor_math_rate']=='null'
    for row in records('category_donor_native_cross.csv'):
        assert sum(int(row[key+'_n']) for key in native_cross)==int(row['category_n'])
    artifact=json.loads((OUT/'artifact_manifest.json').read_text())
    for name,info in artifact.items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==info['sha256'],name
    for name in ['report.md','README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text()):
            assert (OUT/target).exists(),(name,target)
    direct=json.loads((OUT/'direct_index_checks.json').read_text())
    assert direct['n']==600 and direct['all_passed']
    assert direct['max_probability_reduction_error']<=plan['independent_probability_reduction_atol']
    status=json.loads((OUT/'status.json').read_text())
    assert status['status']=='COMPLETED' and status['report_complete'] and not status['report_pending']
    print('PASS: frozen plan/code; membership; all CSV partitions/denominators; six-seed means; '
          'NN empty strata; category/native cross-table; fixed direct reads; report links; artifact hashes.')


if __name__=='__main__':
    main()
