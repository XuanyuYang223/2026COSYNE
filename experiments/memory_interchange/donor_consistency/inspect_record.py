"""Decode a compact observation and retrieve both original probability vectors."""
import argparse
import numpy as np
from .common import *

def inspect(dataset,seed,index):
    plan=load_plan();key=semantic_decode(plan,dataset,seed,index)
    entry=next(e for e in plan['entries'] if all(e[k]==v for k,v in
        [('dataset',dataset),('seed',seed),('d',key['d']),('original_S',key['original_S']),('treatment',key['treatment'])]))
    q=np.array(key['q']);state=np.load(state_path(dataset,seed));patch=np.load(BASE/entry['patch_path']);native=np.load(BASE/entry['native_path'])
    pi=np.flatnonzero(np.all(patch['content']==q,axis=-1));ni=np.flatnonzero(np.all(state['content']==q,axis=-1))
    assert len(pi)==len(ni)==1
    a,b=key['donor_condition_id'],key['recipient_condition_id'];kind=QUERY_TYPES.index(key['query_type']);item=key['query_item']
    lp=patch[entry['patch_field']][a,b,int(pi[0]),kind,item]
    ln=native[entry['native_field']][a,int(ni[0]),kind,item]
    p=probabilities(lp)[0] if entry['patch_probability'] else None;dn=probabilities(ln)[0]
    root=OUT/'data'/dataset/f'seed_{seed}';shape=(6,2,306,len(plan['datasets'][dataset]['members']),2,7)
    idx=np.unravel_index(index,shape);ans=np.load(root/'answers.npy',mmap_mode='r')[idx]
    d=np.load(root/'probability_differences.npy',mmap_mode='r')[idx]
    return {**key,'patch_prediction':int(ans[0]),'donor_prediction':int(ans[1]),'donor_truth':int(ans[2]),
        'donor_native_wrong':bool(ans[1]!=ans[2]),'patch_probability':p.tolist() if p is not None else None,
        'donor_probability':dn.tolist(),'max_abs_probability_difference':float(d[0]) if np.isfinite(d[0]) else None,
        'TV':float(d[1]) if np.isfinite(d[1]) else None,'patch_file':entry['patch_path'],'native_file':entry['native_path']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--flat-index',type=int,required=True);a=p.parse_args()
    print(json.dumps(inspect(a.dataset,a.seed,a.flat_index),ensure_ascii=False,indent=2,allow_nan=False))
