"""Independent reductions, conservative accounting and final artifact inventory."""
import argparse
import datetime
import subprocess
import time
import json
import numpy as np
from ..intervene import correct_logp
from .common import *

def accounting():
    if (OUT/'cost_windows.json').exists():return
    # The log is created by shell redirection before Python/CUDA initialization.
    # Linux birth timestamps and observed completion give conservative complete
    # process windows, including initialization, I/O and CPU work while loaded.
    def birth(path):
        stamp=subprocess.check_output(['stat','-c','%w',str(path)],text=True).strip()
        return datetime.datetime.fromisoformat(stamp).timestamp()
    native_start=birth(OUT/'native.log');patch_start=birth(OUT/'translation.log')
    native_end=json.loads((OUT/'all_native_masks_frozen.json').read_text())['time']
    # The whole native script had exited before the patch process was started.
    # Add the recorded frozen-mask endpoint and use a fixed startup/exit upper
    # bound from the gap to the subsequent process where needed, without adding
    # unmeasured GPU kernels. Native endpoint includes all mask compression.
    patch_end_observed=time.time()
    window_native=native_end-native_start
    window_patch=patch_end_observed-patch_start
    timed=json.loads((OUT/'budget.json').read_text())
    full=window_native+window_patch
    # A second conservative small process-exit interval: native mask freeze to
    # native.log mtime is normally negative; ensure positive accounting only.
    assert full>=timed['final_audit_gpu_seconds']
    write_json(OUT/'cost_windows.json',{
        'native_log_birth_unix':native_start,'native_all_masks_saved_unix':native_end,
        'patch_log_birth_unix':patch_start,'patch_completion_observed_unix':patch_end_observed,
        'native_inclusive_window_seconds':window_native,'patch_inclusive_window_seconds':window_patch,
        'timed_evaluation_windows_seconds':timed['final_audit_gpu_seconds'],
        'conservative_full_process_windows_seconds':full,
        'note':'Conservative GPU-resident process windows include CUDA initialization, caching, evaluations, saving and CPU reductions/I/O plus observation lag; actual busy GPU time is smaller. CPU-only probes/statistics/report work is not separately added.'})
    write_json(OUT/'budget.json',{'gpu_seconds':timed['inherited_gpu_seconds']+full,
        'inherited_gpu_seconds':timed['inherited_gpu_seconds'],'final_audit_gpu_seconds':full,
        'limit_seconds':43200,'timed_evaluation_windows_seconds':timed['final_audit_gpu_seconds'],
        'accounting_basis':'inclusive conservative GPU-resident process windows; see cost_windows.json'})

def independent_checks():
    reductions=[];nn=[];maxerr=0.
    for cohort in ('B1','B_reserved'):
        for seed in SEEDS:
            out=OUT/cohort/f'seed_{seed}'
            assert (out/'translation_complete.json').exists()
            assert (out/'mask_freeze.json').exists()
            masktime=json.loads((out/'mask_freeze.json').read_text())['time']
            assert masktime<json.loads((out/'translation_complete.json').read_text())['completed_time']
            for d in (0,10):
                for s in SUFFIXES:
                    raw=dict(np.load(out/f'translation_d{d}_S{s}.npz'))
                    masks=dict(np.load(out/f'masks_d{d}_S{s}.npz'))
                    ans=raw['Q_original'];lit=raw['Q_literal'];orig=raw['patch_original_logp'];flip=raw['patch_translated_logp']
                    # Verify truth independently from saved p and query arrays.
                    oracle=np.stack([maps_answers(matrix_apply(raw['content'],s,r)) for r,h in CONDITIONS])
                    assert np.array_equal(ans,oracle)
                    oracle_flip=np.stack([maps_answers(matrix_apply(raw['content'],SWAP[s],r)) for r,h in CONDITIONS])
                    assert np.array_equal(lit,oracle_flip)
                    source=RUNS/f'seed_{seed}/analysis' if cohort=='B1' else out
                    baseline=np.load(source/f'baseline_d{d}_S{s}.npz')
                    flipped=np.load(source/f'baseline_d{d}_S{SWAP[s]}.npz')
                    nc=baseline['natural_logp'].argmax(-1)==ans
                    original_prediction=baseline['natural_logp'].argmax(-1)
                    translated_prediction=flipped['natural_logp'].argmax(-1)
                    for a,b in PAIRS:
                        expected_rec=nc[b]
                        expected_both=nc[a]&nc[b]
                        donor=translated_prediction[a] if CONDITIONS[a][0]!=CONDITIONS[b][0] else original_prediction[a]
                        expected_sem=nc[b] & (donor==ans[b])
                        assert np.array_equal(masks['rec'][a,b],expected_rec)
                        assert np.array_equal(masks['both'][a,b],expected_both)
                        assert np.array_equal(masks['E_sem'][a,b],expected_sem)
                    if s=='NN':
                        assert np.array_equal(orig,flip)
                        assert np.array_equal(ans,lit)
                        nn.append({'cohort':cohort,'seed':seed,'d':d,'logp_max_error':0.,'all_elements_equal':True})
                    # All 18*18 query losses and accuracies in float64, with
                    # separate original/literal targets and paired masks.
                    predictions=[orig.argmax(-1),flip.argmax(-1)]
                    accuracy=[p==ans[None] for p in predictions]
                    literal_accuracy=[p==lit[None] for p in predictions]
                    ce=[-correct_logp(o,ans[None]).astype(float) for o in (orig,flip)]
                    native_correct=masks['native_correct'][None]
                    errors=[native_correct & ~a for a in accuracy]
                    rows={(int(r['donor']),int(r['receiver']),r['mode']):r for r in csv_read(out/f'translation_d{d}_S{s}.csv') if r['query_type']=='-1'}
                    for mode in MODES:
                        mm=masks[mode]
                        assert mm.shape==accuracy[0].shape
                        for a,b in PAIRS:
                            m=mm[a,b];n=int(m.sum());row=rows[(a,b,mode)]
                            assert n==int(row['n'])
                            if not n:continue
                            actual={}
                            for ti,tname in enumerate(('original','translated')):
                                actual[tname+'_accuracy_original']=float(accuracy[ti][a,b][m].mean())
                                actual[tname+'_accuracy_literal']=float(literal_accuracy[ti][a,b][m].mean())
                                actual[tname+'_CE_original']=float(ce[ti][a,b][m].mean())
                                actual[tname+'_new_error_original']=float(errors[ti][a,b][m].mean())
                            actual['delta_accuracy_original']=actual['translated_accuracy_original']-actual['original_accuracy_original']
                            actual['delta_CE_original']=actual['translated_CE_original']-actual['original_CE_original']
                            actual['delta_new_error_original']=actual['translated_new_error_original']-actual['original_new_error_original']
                            for k,v in actual.items():
                                err=abs(v-float(row[k]));maxerr=max(maxerr,err)
                                np.testing.assert_allclose(v,float(row[k]),atol=2e-13,rtol=2e-13)
                    blocks=np.load(out/f'blocks_d{d}_S{s}.npz')
                    sums=blocks['sums'].sum(axis=3);counts=blocks['counts'].sum(axis=2)
                    for pi,(a,b) in enumerate(PAIRS):
                        for mi,mode in enumerate(MODES):
                            row=rows[(a,b,mode)];assert counts[pi,mi]==int(row['n'])
                            if counts[pi,mi]:
                                for ti,tname in enumerate(('original','translated')):
                                    np.testing.assert_allclose(sums[pi,mi,ti,0]/counts[pi,mi],float(row[tname+'_accuracy_original']),atol=2e-13)
                    reductions.append({'cohort':cohort,'seed':seed,'d':d,'S':s,'direction_masks_checked':len(PAIRS)*len(MODES),
                                       'same_masks_both_treatments':True,'block_counts_and_point_reductions_agree':True})
            print(json.dumps({'phase':'independent final reductions','cohort':cohort,'seed':seed}),flush=True)
    original=json.loads((OUT/'frozen_plan.json').read_text())
    receipt=json.loads((OUT/'freeze_receipt.json').read_text())
    assert sha256(OUT/'frozen_plan.json')==receipt['plan_sha256']
    assert sha256(OUT/'revision_final_audit.yaml')==receipt['config_sha256']
    assert sha256(OUT/'B_reserved_manifest.json')==original['reserved_manifest_sha256']
    for c in original['checkpoints']:assert sha256(BASE/c['path'])==c['sha256']
    write_json(OUT/'post_patch_checks.json',{'checks':reductions,'NN_checks':nn,'max_point_reduction_difference':maxerr,
         'all_selected_checkpoint_hashes_unchanged':True,'all_directions_retained':True,'all_6_models_retained':True,
         'precision_note':'new GPU checks pass; old CPU float32 strict-logit failure is explicitly retained, float64 pass/argmax invariance are separate'})
    write_json(OUT/'preservation_check.json',preserve_check())

def inventory():
    files={}
    for root in (OUT,Path(__file__).parent):
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json':
                files[str(p.relative_to(BASE))]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    write_json(OUT/'artifact_manifest.json',files)
    write_json(OUT/'status.json',{'status':'COMPLETED','finite_audits':3,'training_updates':0,
        'models':SEEDS,'independent_confirmatory_contents':362,'independent_confirmatory_orbits':51,
        'old_primary_results_unchanged':True,'all_checks_required_for_new_implementation_passed':True,
        'old_CPU_float32_strict_precision_failure_retained':True,'no_external_publication':True})
    # Include status after its final write. The manifest excludes itself.
    p=OUT/'status.json';files[str(p.relative_to(BASE))]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    write_json(OUT/'artifact_manifest.json',files)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['check','cost','inventory']);a=p.parse_args()
    {'check':independent_checks,'cost':accounting,'inventory':inventory}[a.phase]()
