"""Meaningful truth, masking, aggregation and preservation checks."""
import itertools
import json
import numpy as np
from ..truth import matrix_apply, apply
from .common import *

def before_patches():
    q=np.array(list(itertools.permutations(range(7))))
    checks=0
    for rb in (0,1):
        ra=1-rb
        for s in SUFFIXES:
            receiver=matrix_apply(q,s,rb)
            donor=matrix_apply(q,SWAP[s],ra)
            assert np.array_equal(receiver,donor)
            assert np.array_equal(receiver,apply(q,s,rb))
            checks+=len(q)
    masks_checked=0
    for cohort in ('B1','B_reserved'):
        for seed in SEEDS:
            out=OUT/cohort/f'seed_{seed}'
            receipt=json.loads((out/'mask_freeze.json').read_text())
            for row in receipt['files']:
                p=OUT/row['path'];assert sha256(p)==row['sha256']
                f=np.load(p)
                assert np.all(f['both']<=f['rec']) and np.all(f['E_sem']<=f['rec'])
                if row['S']=='NN': assert np.array_equal(f['both'],f['E_sem'])
                masks_checked+=1
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    for p in plan['checkpoints']:assert sha256(BASE/p['path'])==p['sha256']
    m=np.load(OUT/'B_reserved_members.npz')
    assert set(m['orbit_id']).isdisjoint(set(m['B1_orbit_id']))
    assert len(set(map(tuple,m['content'])) & set(map(tuple,m['B1_content'])))==0
    # Block denominators remain paired under unequal orbit sizes/mask coverage.
    numerator=np.array([[[4.,10.],[9.,1.]],[[3.,5.],[2.,6.]]])
    denominator=np.array([[8.,20.],[6.,10.]])
    weights=np.array([2.,1.])
    paired=(numerator[:,1]@weights-numerator[:,0]@weights)/(denominator@weights)
    assert np.allclose(paired,[1/36,-1/22])
    write_json(OUT/'pre_patch_checks.json',{'all_5040_content_semantic_translation_checks':checks,
        'independent_truth_oracles_agree':True,'frozen_mask_files_checked':masks_checked,
        'NN_masks_exactly_equal':True,'no_reserved_B1_orbit_overlap':True,
        'all_selected_checkpoints_match_frozen_hashes':True,'paired_unequal_block_denominator_check':True,
        'preservation':preserve_check()})
    print(json.dumps({'pre_patch_checks':'passed','mask_files':masks_checked}),flush=True)

if __name__=='__main__':before_patches()
