import itertools
import json
import time
import numpy as np
import torch
from ..cache import cache_states
from ..common import config as original_config,seed_all,write_json
from ..geometry import linear_cka,raw_distance,procrustes_distance,ridge_fit,ridge_apply
from ..predict import save_csv
from ..train import load_model
from ..truth import contents
from .common import BASE,RESULTS,RUNS,Budget
from .intervene import evaluate

def dual_probes(model,manifest,cfg,out):
    q,ids=contents(manifest,'T')
    index=np.random.default_rng(cfg['analysis']['probe_seed']).choice(len(q),512,replace=False)
    cache=cache_states(model,q[index],ids[index],cfg,out/'probe_T_states.npz')
    x=cache['memory'].reshape(-1,model.memory_dim)
    current_y=np.tile(np.eye(7)[cache['content']].reshape(512,49),(36,1))
    initial_y=np.eye(7)[cache['initial']].reshape(-1,49)
    current=ridge_fit(x,current_y,1.);initial=ridge_fit(x,initial_y,1.)
    assert np.array_equal(current['x_mean'],initial['x_mean'])
    np.savez_compressed(out/'current_probe.npz',**current,content=q[index],orbit_id=ids[index])
    np.savez_compressed(out/'initial_probe.npz',**initial,content=q[index],orbit_id=ids[index])
    return current,initial

def probe_scores(cache,fit,target):
    prediction=ridge_apply(cache['memory'],fit).reshape(*cache['memory'].shape[:-1],7,7).argmax(-1)
    correct=prediction==target
    return correct.mean(axis=(-1,-2)),correct.all(-1).mean(-1)

def measure(checkpoint,initial_checkpoint,out,seed,exploratory=False):
    out.mkdir(parents=True,exist_ok=True)
    cfg=original_config();manifest=json.loads((BASE/'split_manifest.json').read_text())
    seed_all(123,4,True)
    with Budget() as budget:
        budget.check(120);model,saved=load_model(checkpoint)
        assert saved['seed']==seed
        if exploratory:
            subset=np.load(RESULTS/'V_subset_manifest.npz')
            g,gi=subset['geometry_content'],subset['geometry_orbit_id']
            b,bi=subset['behavior_content'],subset['behavior_orbit_id']
        else:
            assert seed in cfg['seeds']
            g,gi=contents(manifest,'G');b,bi=contents(manifest,'B',behavior=True)
            assert len(b)==134
        if not (out/'intervention_summary.json').exists():
            behavior=cache_states(model,b,bi,cfg,out/'behavior_states.npz')
            evaluate(model,behavior,cfg,out,seed,exploratory)
        if not (out/'geometry_summary.json').exists():
            started=time.perf_counter()
            geometry=cache_states(model,g,gi,cfg,out/'geometry_states.npz')
            current_probe,initial_probe=dual_probes(model,manifest,cfg,out)
            initial_model,_=load_model(initial_checkpoint)
            init=cache_states(initial_model,g,gi,cfg,out/'initial_geometry_states.npz')
            item,whole=probe_scores(geometry,current_probe,geometry['content'][None,None])
            item_initial,whole_initial=probe_scores(geometry,initial_probe,geometry['initial'])
            rows=[]
            for di,d in enumerate(cfg['timing']['delays']):
                for a,b in itertools.combinations(range(18),2):
                    x,y=geometry['memory'][di,a],geometry['memory'][di,b]
                    pa=np.eye(7)[geometry['initial'][di,a]].reshape(len(g),49)
                    pb=np.eye(7)[geometry['initial'][di,b]].reshape(len(g),49)
                    rows.append({'seed':seed,'a':a,'b':b,'d':d,'cka':linear_cka(x,y),'raw':raw_distance(x,y),
                                 'proc':procrustes_distance(x,y),'init_cka':linear_cka(init['memory'][di,a],init['memory'][di,b]),
                                 'D_input':raw_distance(pa,pb),
                                 'control_distance':raw_distance(geometry['control'][di,a:a+1],geometry['control'][di,b:b+1]),
                                 'probe_item_acc':float((item[di,a]+item[di,b])/2),
                                 'probe_perm_acc':float((whole[di,a]+whole[di,b])/2),
                                 'probe_initial_item_acc':float((item_initial[di,a]+item_initial[di,b])/2),
                                 'probe_initial_perm_acc':float((whole_initial[di,a]+whole_initial[di,b])/2)})
            save_csv(out/'geometry.csv',rows)
            write_json(out/'geometry_summary.json',{'seed':seed,'exploratory':exploratory,'contents':len(g),
                       'current_probe_item_accuracy':item.tolist(),'current_probe_full_accuracy':whole.tolist(),
                       'initial_probe_item_accuracy':item_initial.tolist(),'initial_probe_full_accuracy':whole_initial.tolist(),
                       'probe_same_T_inputs_same_centering_alpha1':True,'elapsed_seconds':time.perf_counter()-started})

def exploratory():
    lock=json.loads((RESULTS/'checkpoint_lock.json').read_text())
    measure(RUNS/'pilot_seed0/selected.pt',BASE/'runs/v1_2/candidate_A_seed0/initial.pt',RESULTS/'V_exploration',0,True)

if __name__=='__main__':exploratory()
