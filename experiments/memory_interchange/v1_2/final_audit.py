"""Read-only checkpoint checks and matched T/V state paths at diagnostic points."""
import json
import numpy as np
import torch
from ..common import rng_state, write_json, sha256
from ..v1_1.audit import subset
from ..v1_1.data import load
from ..v1_1.trace import gradient_path, gate_and_sensitivity
from .common import BASE, RESULTS, RUNS, Budget, config
from .prepare import same_rng
from .train import initialize, optimizer_for, restore_checkpoint

def augment():
    out=RESULTS/'recovery_diagnostics';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    revision=config();pilot=RUNS/'candidate_A_seed0'
    points=[('restore_stage6_step5000',BASE/revision['resume']['checkpoint'],6),
            ('stage6_endpoint',pilot/'stage_6'/'advance.pt',6),
            ('stage7_endpoint',pilot/'stage_7'/'latest.pt',7)]
    result={}
    with Budget():
        for label,path,sid in points:
            if not path.exists():continue
            model=initialize(revision,'A');optimizer=optimizer_for(model,revision)
            rng=np.random.default_rng(100000)
            saved=restore_checkpoint(path,model,optimizer,rng)
            assert same_rng(saved['rng'],rng_state(rng))
            actual=optimizer.state_dict();expected=saved['optimizer']
            assert actual['param_groups']==expected['param_groups']
            for k,state in expected['state'].items():
                for field,v in state.items():
                    a=actual['state'][k][field]
                    assert torch.equal(a.cpu(),v.cpu()) if torch.is_tensor(v) else a==v
            for k,v in saved['model'].items():assert torch.equal(model.state_dict()[k].cpu(),v.cpu())
            record={'checkpoint':str(path.resolve()),'sha256':sha256(path),'stage':sid,
                    'stage_step':saved['step'],'curriculum_updates':saved['total_updates'],
                    'weights_exact':True,'all_Adam_states_and_parameter_groups_exact':True,
                    'all_RNG_exact_before_diagnostics':True}
            for split in ('T','V'):
                small=subset(load(f'stage{sid}_{split}'),revision['validation']['diagnostic_batch'])
                record[split]={'fixed_first_rows':len(small['target']),
                               'gradient_path':gradient_path(model,small,out/f'{label}_{split}_gradient.npz'),
                               'content_gates':gate_and_sensitivity(model,small)}
            result[label]=record
    write_json(out/'summary.json',result)
