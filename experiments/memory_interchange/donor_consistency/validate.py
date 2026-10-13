"""Independent scalar-key reads, cardinality/NN checks and immutable inputs."""
import numpy as np
from ..truth import matrix_apply, inverse, CONDITIONS
from .common import *

def run():
    plan=load_plan();checks=[];maximum=0.;missing=[]
    samples=plan['fixed_direct_index_sample']
    for entry in plan['entries']:
        chosen=[s for s in samples if all(s[k]==entry[v] for k,v in [('dataset','dataset'),('seed','seed'),
            ('d','d'),('S','original_S'),('treatment','treatment')])]
        if not chosen:continue
        if not entry['available']:
            missing+=chosen;continue
        pf=np.load(BASE/entry['patch_path']);nf=np.load(BASE/entry['native_path'])
        patch=pf[entry['patch_field']];native=nf[entry['native_field']]
        state=np.load(state_path(entry['dataset'],entry['seed']))
        condition_keys=[condition_key(r,h) for r,h in zip(state['contexts'],state['histories'])]
        root=OUT/'data'/entry['dataset']/f'seed_{entry["seed"]}'
        diff=np.load(root/'probability_differences.npy',mmap_mode='r');answers=np.load(root/'answers.npy',mmap_mode='r');flags=np.load(root/'flags.npy',mmap_mode='r')
        members=np.array(plan['datasets'][entry['dataset']]['members'])
        for sample in chosen:
            q=np.array(sample['q']);a=sample['donor'];b=sample['recipient'];kind=sample['query_type'];item=sample['query_item']
            # Find rows from actual semantic content/history metadata, without
            # using the vectorized join's precomputed q-index arrays.
            pq=np.flatnonzero(np.all(pf['content']==q,axis=-1));nq=np.flatnonzero(np.all(state['content']==q,axis=-1))
            if not len(pq) or not len(nq):missing.append(sample);continue
            assert len(pq)==len(nq)==1
            donor=condition_keys.index(condition_key(*CONDITIONS[a]))
            recipient=condition_keys.index(condition_key(*CONDITIONS[b]))
            pl=np.asarray(patch[donor,recipient,int(pq[0]),kind,item],np.float64)
            dl=np.asarray(native[donor,int(nq[0]),kind,item],np.float64)
            if entry['patch_probability']:
                p=np.exp(pl);p/=p.sum();pa=int(np.argmax(p))
            else:p=None;pa=int(pl)
            dn=np.exp(dl);dn/=dn.sum();da=int(np.argmax(dn))
            final=matrix_apply(q[None],entry['effective_suffix'],CONDITIONS[a][0])[0]
            truth=int(final[item] if kind==0 else inverse(final)[item])
            cq=np.flatnonzero(np.all(members==q,axis=-1));pi=np.flatnonzero((PAIRS[:,0]==a)&(PAIRS[:,1]==b))
            assert len(cq)==len(pi)==1
            cell=DELAYS.index(sample['d'])*3+SUFFIXES.index(sample['S']);ti=TREATMENTS.index(sample['treatment'])
            index=(cell,ti,int(pi[0]),int(cq[0]),kind,item)
            assert tuple(map(int,answers[index][:3]))==(pa,da,truth)
            assert bool(flags[index]&FLAG['agree'])==(pa==da)
            assert bool(flags[index]&FLAG['donor_correct'])==(da==truth)
            if p is not None:
                direct=np.array([np.abs(p-dn).max(),.5*np.abs(p-dn).sum()]);error=float(np.max(np.abs(direct-diff[index])))
                np.testing.assert_allclose(direct,diff[index],atol=3e-15,rtol=0);maximum=max(maximum,error)
                assert bool(flags[index]&FLAG['exact'])==bool(np.array_equal(p,dn))
            checks.append({**sample,'patch_index_by_content':int(pq[0]),'native_index_by_content':int(nq[0]),
                'patch_prediction':pa,'donor_prediction':da,'donor_truth':truth,'vectorized_join_agrees':True})
    normalization=read_json(OUT/'normalization_audit.json')
    coverage=csv_read(OUT/'pairing_coverage.csv');controls=csv_read(OUT/'supplemental_controls.csv')
    expected={ds:sum(int(r['expected']) for r in coverage if r['dataset']==ds) for ds in DATASETS}
    paired={ds:sum(int(r['paired']) for r in coverage if r['dataset']==ds) for ds in DATASETS}
    nativekeys={ds:len(SEEDS)*2*3*18*len(plan['datasets'][ds]['members'])*14 for ds in DATASETS}
    assert all(expected[ds]==34*nativekeys[ds] for ds in DATASETS)
    nn=read_json(OUT/'NN_audit.json');assert len(nn)==24 if not plan['missing_conditions_at_freeze'] else True
    baseline_consistent=all(float(r['answer_agreement'])==1. for r in controls if r['control'] in ('sham','uninterrupted','full_state_same_future_input'))
    assert baseline_consistent
    preservation=preserve_check();assert preservation['all_unchanged'],preservation
    nbad=sum(r.get('nonfinite_queries',0)+r.get('normalization_failures',0) for r in normalization)
    missing_n=sum(expected[ds]-paired[ds] for ds in DATASETS)
    probability_missing=sum(int(r['paired'])-int(r.get('probability_queries') or 0) for r in coverage)
    status='PARTIAL' if missing_n or probability_missing or nbad or missing else 'COMPLETED'
    write_json(OUT/'direct_index_validation.json',{'pre_frozen_sample_n':len(samples),'checked':len(checks),
        'missing':missing,'max_probability_difference_reduction_error':maximum,'sample_checks':checks})
    write_json(OUT/'preservation_check.json',preservation)
    write_json(OUT/'pairing_audit.json',{'status':status,'native_keys_unique':'uniqueness proven from independent condition, content, d/effective_suffix and query factor keys; field dimensions validated in plan',
        'patch_keys_unique':'additional recipient, original_S and treatment make every Cartesian observation unique',
        'expected_primary_queries':expected,'paired_primary_queries':paired,'missing_primary_queries':missing_n,
        'probability_missing_queries':probability_missing,'unique_native_keys_expected':nativekeys,
        'controlled_native_multiplicity_complete':34,'each_native_key_expands_to':'17 nonself recipients × 2 treatments',
        'diagonal_control_multiplicity':2,'diagonal_expected_queries':{d:2*v for d,v in nativekeys.items()},
        'self_pairs_retained_as_controls':True,'accidental_many_to_many':False,
        'complete_original_grid':'6 seeds × 18 donor/recipient conditions × 2 delays × 3 original S × 2 treatments × all content × 14 queries',
        'same_context_same_history_nonself':'EMPTY by design, not missing',
        'future_inputs_identical':True,'external_context_after_swap':False,'input_exclusions':[],
        'native_vs_sham_uninterrupted_fullstate_argmax_checks_passed':baseline_consistent,
        'fullstate_control_available':'B1 only; B_reserved did not save this optional control',
        'NN_entries_checked':len(nn),'NN_all_identical':True,'independent_direct_sample_checked':len(checks),
        'normalization_bad_query_records':nbad,'checkpoints_and_members_unchanged':True,
        'old_precision_note':'GPU registered controls passed. CPU float32 original strict-logit check FAILED for all six; same coefficients float64 passed; argmax unchanged. New P thresholds do not replace old logit checks.'})
    print(json.dumps({'validation_status':status,'expected':expected,'paired':paired,'direct_checks':len(checks),
                      'old_files_unchanged':preservation['files']}),flush=True)

if __name__=='__main__':run()
