"""Separate refitting-algorithm instability from fixed LOSO prediction scores."""
from collections import Counter
import json
import numpy as np
from ..predict import fit_predict, transform, errors, segment
from ..v1_3.predict import merged_rows, design, bootstrap_labels
from .common import *

FEATURES=['d','S_NN','S_NA','S_NB','cross_context','native_ce','current_probe_item_acc',
          'control_distance','non_N_Ha','non_N_Hb','same_last_cue','r_a_0','r_a_1','r_b_0','r_b_1','D_input','cka']
MODELS=('M0_v13','M_CKA_v13')

def fixed_predictions():
    original=csv_read(OLD/'prediction/predictions.csv');saved_scores=csv_read(OLD/'prediction/scores.csv')
    folds=[]
    for seed in SEEDS:
        selected=[r for r in original if r['mode']=='all' and int(r['seed'])==seed and r['segment']=='unseen_cross']
        y=np.array([float(r['observed']) for r in selected])
        score={m:errors(y,np.array([float(r[m]) for r in selected])) for m in MODELS}
        old=[r for r in saved_scores if r['seed']==str(seed) and r['mode']=='all' and
             r['segment']=='unseen_cross' and r['quality']=='all' and r['model']=='M_CKA_v13'][0]
        delta=score['M0_v13']['mae']-score['M_CKA_v13']['mae']
        np.testing.assert_allclose(delta,float(old['mae_improvement']),atol=1e-13)
        folds.append({'seed':seed,'n_pairs_d_S':len(selected),
                      'M0_MAE':score['M0_v13']['mae'],'M_CKA_MAE':score['M_CKA_v13']['mae'],
                      'paired_MAE_improvement':delta,
                      'paired_RMSE_improvement':score['M0_v13']['rmse']-score['M_CKA_v13']['rmse']})
    csv_write(OUT/'fixed_prediction_fold_scores.csv',folds)
    values=np.array([r['paired_MAE_improvement'] for r in folds])
    rng=np.random.default_rng(20261017);idx=rng.integers(6,size=(100000,6))
    boot=values[idx].mean(-1)
    np.savez_compressed(OUT/'fixed_prediction_fold_bootstrap.npz',samples=boot,fold_indices=idx,fold_scores=values,seed=20261017)
    summary={'folds':folds,'mean_MAE_improvement':float(values.mean()),'range':values[[values.argmin(),values.argmax()]].tolist(),
        'positive_seeds':int((values>0).sum()),'negative_seeds':int((values<0).sum()),'zero_seeds':int((values==0).sum()),
        'conditional_empirical_95ci':np.quantile(boot,[.025,.975]).tolist(),'replicates':100000,
        'fixed_objects':'original six LOSO fitters and original test predictions/labels',
        'conditioning':'six observed trained networks, predictors and original test data; no training/refitting uncertainty or unknown-model population inference',
        'original_prediction_sha256':sha256(OLD/'prediction/predictions.csv'),
        'not_replacement_for_refitting_bootstrap':True}
    write_json(OUT/'fixed_prediction_summary.json',summary)

def replay():
    plan=json.loads((OUT/'frozen_plan.json').read_text())['bootstrap_replay']
    reps=plan['replicate_ids'];rows=merged_rows([RUNS/f'seed_{s}' for s in SEEDS])
    by={s:[r for r in rows if int(r['seed'])==s] for s in SEEDS}
    blocks={s:dict(np.load(RUNS/f'seed_{s}/analysis/orbit_blocks.npz')) for s in SEEDS}
    xs={s:{m:design(by[s],m) for m in MODELS} for s in SEEDS}
    seen={s:np.array([not r['unseen_pair'] for r in by[s]]) for s in SEEDS}
    segments={s:np.array([segment(r) for r in by[s]]) for s in SEEDS}
    original={(int(r['replicate']),r['mode']):r for r in csv_read(OLD/'prediction/bootstrap.csv')
              if r['segment']=='unseen_cross' and r['model']=='M_CKA_v13'}
    rng=np.random.default_rng(20261016);metadata=[];features=[];scores=[]
    out=OUT/'bootstrap_replay';out.mkdir(exist_ok=True)
    for rep in range(max(reps)+1):
        selected=rng.choice(SEEDS,6,replace=True)
        if len(set(selected))<2:continue
        weights={s:np.bincount(rng.integers(len(blocks[s]['orbit_ids']),size=len(blocks[s]['orbit_ids'])),
                                  minlength=len(blocks[s]['orbit_ids'])) for s in set(selected)}
        if rep not in reps:continue
        for mi,mode in enumerate(('all','rec','both')):
            ys={s:bootstrap_labels(by[s],blocks[s],weights[s],mi) for s in set(selected)}
            prediction={}
            for held in sorted(set(selected)):
                fitted=[int(s) for s in selected if s!=held]
                assert held not in fitted
                masks={s:seen[s]&np.isfinite(ys[s]) for s in fitted}
                yt=np.concatenate([ys[s][masks[s]] for s in fitted]);prediction[held]={}
                fit_units=[(s,int(i)) for s in fitted for i in np.flatnonzero(masks[s])]
                signature=digest_array(np.array(fit_units,np.int64));label_sha=digest_array(yt)
                for model in MODELS:
                    x=np.concatenate([xs[s][model][masks[s]] for s in fitted]);xt=xs[held][model]
                    p,prep,fit=fit_predict(x,yt,xt,1.);prediction[held][model]=p
                    z=transform(xt,prep);where=int(np.argmax(np.abs(p)))
                    maximum=int(np.argmax(p));sample=by[held][where]
                    control=x[:,7];test_control=xt[:,7]
                    entry={'replicate':rep,'mode':mode,'model':model,'selected_seeds':selected.tolist(),
                        'held_seed':int(held),'fit_seed_copies':fitted,'fit_seed_weights':dict(Counter(fitted)),
                        'unique_fitting_models':len(set(fitted)),'held_seed_overlap':False,'fitting_rows':len(yt),
                        'fit_units_sha256':signature,'fit_labels_sha256':label_sha,
                        'shared_paired_resampling_labels_test_units':True,
                        'test_rows':len(p),'maximum_prediction':float(p[maximum]),'maximum_absolute_prediction':float(abs(p[where])),
                        'maximum_absolute_sample':{k:(int(sample[k]) if k!='S' else sample[k]) for k in ('seed','a','b','d','S')},
                        'all_predictions_finite':bool(np.isfinite(p).all()),'max_test_abs_z':float(np.abs(z).max()),
                        'ridge_alpha':1.,'ridge_weights':fit['weights'].ravel().tolist(),
                        'fit_x_mean':fit['x_mean'].tolist(),'fit_y_mean':fit['y_mean'].tolist(),
                        'control_raw_train_min':float(control.min()),'control_raw_train_max':float(control.max()),
                        'control_raw_train_std':float(control.std()),'control_raw_test_min':float(test_control.min()),
                        'control_raw_test_max':float(test_control.max()),'control_raw_scale':float(prep['scale'][7])}
                    metadata.append(entry)
                    for j,name in enumerate(FEATURES[:x.shape[1]]):
                        features.append({'replicate':rep,'mode':mode,'model':model,'held_seed':int(held),'feature_index':j,
                            'feature':name,'imputation_median':float(prep['median'][j]),'fit_mean':float(prep['mean'][j]),
                            'fit_scale':float(prep['scale'][j]),'raw_train_min':float(np.nanmin(x[:,j])),
                            'raw_train_max':float(np.nanmax(x[:,j])),'raw_train_std':float(np.nanstd(x[:,j])),
                            'raw_test_min':float(np.nanmin(xt[:,j])),'raw_test_max':float(np.nanmax(xt[:,j])),
                            'test_z_min':float(z[:,j].min()),'test_z_max':float(z[:,j].max()),
                            'ridge_coefficient':float(fit['weights'][j,0])})
                    np.savez_compressed(out/f'rep{rep}_{mode}_held{held}_{model}.npz',prediction=p,fit_labels=yt,
                        fit_units=np.array(fit_units),selected_seeds=selected,**prep,
                        **{'fit_'+k:v for k,v in fit.items()},
                        **{f'orbit_weights_{s}':weights[s] for s in weights})
                # Independently assert the two models received identical label/unit arrays.
                m0=metadata[-2];m1=metadata[-1]
                assert m0['fit_units_sha256']==m1['fit_units_sha256'] and m0['fit_labels_sha256']==m1['fit_labels_sha256']
            yy=np.concatenate([ys[int(s)][segments[int(s)]=='unseen_cross'] for s in selected])
            pp={m:np.concatenate([prediction[int(s)][m][segments[int(s)]=='unseen_cross'] for s in selected]) for m in MODELS}
            valid=np.isfinite(yy);yy=yy[valid]
            err={m:errors(yy,p[valid]) for m,p in pp.items()}
            delta={k+'_improvement':err['M0_v13'][k]-err['M_CKA_v13'][k] for k in ('mae','rmse')}
            old=original[(rep,mode)]
            for k,v in delta.items():np.testing.assert_allclose(v,float(old[k]),atol=2e-7,rtol=2e-10)
            scores.append({'replicate':rep,'mode':mode,'n_test_cells_with_seed_copies':len(yy),**delta,
                'MAE_replay_minus_saved':delta['mae_improvement']-float(old['mae_improvement']),
                'RMSE_replay_minus_saved':delta['rmse_improvement']-float(old['rmse_improvement'])})
        print(json.dumps({'phase':'bootstrap replay','replicate':rep}),flush=True)
    csv_write(OUT/'bootstrap_feature_diagnostics.csv',features);csv_write(OUT/'bootstrap_replay_scores.csv',scores)
    write_json(OUT/'bootstrap_replay_metadata.json',metadata)
    write_json(OUT/'bootstrap_replay_summary.json',{'replicate_ids':reps,'typical':plan['typical'],
        'minimum':plan['minimum'],'maximum':plan['maximum'],'fold_fits':len(metadata),
        'all_original_seed_copies_excluded':True,'paired_same_resampling_labels_test_units':True,
        'all_predictions_finite':all(r['all_predictions_finite'] for r in metadata),
        'max_absolute_prediction':max(r['maximum_absolute_prediction'] for r in metadata),
        'max_test_abs_z':max(r['max_test_abs_z'] for r in metadata),
        'max_MAE_replay_difference':max(abs(r['MAE_replay_minus_saved']) for r in scores),
        'implementation_error_found':False,'large_extrapolation_is_not_automatically_a_bug':True,
        'no_clipping_no_alpha_change_no_variance_floor_change':True,
        'original_bootstrap_sha256':sha256(OLD/'prediction/bootstrap.csv'),
        'original_intervals_sha256':sha256(OLD/'prediction/bootstrap_intervals.json')})

if __name__=='__main__':
    fixed_predictions();replay()
