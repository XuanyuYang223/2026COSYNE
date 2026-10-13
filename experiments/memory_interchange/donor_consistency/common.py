from pathlib import Path
import csv
import json
import hashlib
import zipfile
import itertools
import numpy as np

BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
FINAL=BASE/'Final_audit'
RUNS=BASE/'runs/v1_3'
SEEDS=tuple(range(101,107))
DATASETS=('B1','B_reserved')
SUFFIXES=('NN','NA','NB')
SWAP={'NN':'NN','NA':'NB','NB':'NA'}
TREATMENTS=('original','translated')
DELAYS=(0,10)
QUERY_TYPES=('item','position')
PAIRS=np.array([(a,b) for a in range(18) for b in range(18) if a!=b],np.int16)
FLAG={'available':1,'probability_available':2,'agree':4,'donor_correct':8,'exact':16,
      'patch_tie':32,'donor_tie':64,'close_1e6':128,'close_1e5':256,'close_1e4':512,
      'patch_tie_known':1024,'donor_tie_known':2048}
PERM_ID={p:i for i,p in enumerate(itertools.permutations(range(7)))}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read_json(path):return json.loads(Path(path).read_text())
def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def headers(path):
    result={}
    if not Path(path).exists():return result
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):reader=np.lib.format.read_array_header_1_0
                elif version==(2,0):reader=np.lib.format.read_array_header_2_0
                else:raise ValueError(f'Unsupported NPY header version {version}: {path}/{name}')
                shape,order,dtype=reader(f)
            result[name[:-4]]={'shape':list(shape),'dtype':str(dtype),'fortran':order}
    return result

def record(path):
    path=Path(path)
    return {'path':str(path.relative_to(BASE)),'exists':path.exists(),
            'sha256':sha(path) if path.exists() else None,
            'bytes':path.stat().st_size if path.exists() else 0,
            'fields':headers(path) if path.suffix=='.npz' and path.exists() else {}}

def member_ids(q):return np.array([PERM_ID[tuple(map(int,x))] for x in q],np.int16)

def load_plan():
    plan=read_json(OUT/'analysis_plan.json');receipt=read_json(OUT/'freeze_receipt.json')
    assert sha(OUT/'analysis_plan.json')==receipt['analysis_plan_sha256']
    for name,value in plan['code_sha256'].items():assert sha(BASE/name)==value,(name,'code modified after freeze')
    return plan

def state_path(dataset,seed):
    return RUNS/f'seed_{seed}/analysis/behavior_states.npz' if dataset=='B1' else FINAL/dataset/f'seed_{seed}/states.npz'

def native_path(dataset,seed,d,s):
    root=RUNS/f'seed_{seed}/analysis' if dataset=='B1' else FINAL/dataset/f'seed_{seed}'
    return root/f'baseline_d{d}_S{s}.npz'

def patch_path(dataset,seed,d,s):return FINAL/dataset/f'seed_{seed}'/f'translation_d{d}_S{s}.npz'

def probabilities(logp):
    lp=np.asarray(logp,np.float64)
    finite=np.isfinite(lp).all(-1)
    raw=np.exp(lp);sums=raw.sum(-1)
    normalized=finite & np.isfinite(sums) & (sums>0) & (np.abs(sums-1)<=2e-6)
    p=np.divide(raw,sums[...,None],out=np.full_like(raw,np.nan),where=normalized[...,None])
    if normalized.any():np.testing.assert_allclose(p[normalized].sum(-1),1.,atol=3e-15,rtol=0.)
    return p,normalized,{'nonfinite_queries':int((~finite).sum()),
        'normalization_failures':int((finite & ~normalized).sum()),
        'max_raw_normalization_error':float(np.max(np.abs(sums[finite]-1))) if finite.any() else None}

def condition_key(r,h):return f'r={int(r)};H={str(h)}'

def dataset_meta(plan,dataset):return plan['datasets'][dataset]

def semantic_decode(plan,dataset,seed,flat_index):
    meta=dataset_meta(plan,dataset);shape=(6,2,306,len(meta['members']),2,7)
    cell,ti,pi,qi,kind,item=map(int,np.unravel_index(int(flat_index),shape))
    a,b=map(int,PAIRS[pi]);d=DELAYS[cell//3];s=SUFFIXES[cell%3]
    effective=s if ti==0 else SWAP[s]
    return {'dataset':dataset,'seed':int(seed),'treatment':TREATMENTS[ti],
        'donor_condition':plan['conditions'][a]['key'],'recipient_condition':plan['conditions'][b]['key'],
        'donor_condition_id':a,'recipient_condition_id':b,'content_id':meta['content_ids'][qi],
        'q':meta['members'][qi],'orbit_id':meta['orbit_ids_per_content'][qi],
        'd':d,'original_S':s,'effective_suffix':effective,'query_type':QUERY_TYPES[kind],'query_item':item,
        'swap_event_step':21+d,'query_first_step':35+d,'query_final_step':37+d}

def preserve_check():
    old=read_json(OUT/'preservation_manifest.json')
    bad=[name for name,v in old.items() if not (BASE/name).exists() or sha(BASE/name)!=v['sha256']]
    return {'files':len(old),'mismatches':bad,'all_unchanged':not bad}
