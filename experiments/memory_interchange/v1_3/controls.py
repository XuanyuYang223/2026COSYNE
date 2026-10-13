import numpy as np
import torch
from ..common import write_json
from ..intervene import continue_pairs
from ..model import MemoryControl
from ..task import trials,tensor_inputs,suffix_inputs,prefix_inputs
from ..truth import CONDITIONS

@torch.no_grad()
def natural_outputs(model,cache,di,suffix,cfg):
    n=cfg['n'];q=cache['content'];dev=next(model.parameters()).device
    result=[]
    for a,(r,h) in enumerate(CONDITIONS):
        p=np.repeat(cache['initial'][di,a],2*n,axis=0)
        data=trials(p,np.full(len(p),r),np.array([h+suffix]*len(p)),np.full(len(p),cache['delays'][di]),
                    np.tile(np.repeat([0,1],n),len(q)),np.tile(np.arange(n),2*len(q)),cfg)
        outputs=[]
        for start in range(0,len(p),cfg['execution']['evaluation_batch_size']):
            part={k:v[start:start+cfg['execution']['evaluation_batch_size']] for k,v in data.items()}
            logits=model(tensor_inputs(part,dev),torch.as_tensor(part['query'],device=dev))
            outputs.append(logits.log_softmax(-1).cpu().numpy())
        result.append(np.concatenate(outputs).reshape(len(q),2,n,n))
    return np.array(result)

@torch.no_grad()
def cpu_strict_control(model,cache,cfg):
    cpu=MemoryControl(cfg);cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()});cpu.eval()
    from ..cache import cache_states
    q=cache['content'][:3];small=cache_states(cpu,q,cache['orbit_id'][:3],cfg)
    kind=np.array([0,1,0]);item=np.array([0,4,6]);maximum=0.;failed=[];prediction_changes=0
    for di,d in enumerate(cfg['timing']['delays']):
        for a,(r,h) in enumerate(CONDITIONS):
            future=suffix_inputs(['NA']*3,kind,item,cfg)
            cont=cpu(tensor_inputs(future,torch.device('cpu')),torch.from_numpy(future['query']),
                     torch.from_numpy(small['memory'][di,a]),torch.from_numpy(small['control'][di,a]).expand(3,-1))
            full=trials(small['initial'][di,a],np.full(3,r),np.array([h+'NA']*3),np.full(3,d),kind,item,cfg)
            natural=cpu(tensor_inputs(full,torch.device('cpu')),torch.from_numpy(full['query']))
            try:torch.testing.assert_close(cont,natural,atol=2e-7,rtol=2e-6)
            except AssertionError:failed.append({'d':d,'condition':a,'max_logit_error':float((cont-natural).abs().max())})
            prediction_changes+=int((cont.argmax(-1)!=natural.argmax(-1)).sum())
            maximum=max(maximum,float((cont-natural).abs().max()))
    # Identical learned float32 coefficients represented exactly in float64.
    # No threshold, scientific state, checkpoint, or training computation changes.
    cpu.double();maximum64=0.
    def inputs(data):return {k:v.double() if v.is_floating_point() else v for k,v in tensor_inputs(data,torch.device('cpu')).items()}
    for di,d in enumerate(cfg['timing']['delays']):
        for a,(r,h) in enumerate(CONDITIONS):
            pref=prefix_inputs(small['initial'][di,a],np.full(3,r),[h]*3,np.full(3,d),cfg)
            m,c=cpu.states(inputs(pref))
            future=suffix_inputs(['NA']*3,kind,item,cfg)
            cont=cpu(inputs(future),torch.from_numpy(future['query']).double(),m,c)
            full=trials(small['initial'][di,a],np.full(3,r),np.array([h+'NA']*3),np.full(3,d),kind,item,cfg)
            natural=cpu(inputs(full),torch.from_numpy(full['query']).double())
            torch.testing.assert_close(cont,natural,atol=2e-7,rtol=2e-6)
            maximum64=max(maximum64,float((cont-natural).abs().max()))
    return {'float32_max_logit_error':maximum,'float32_strict_tolerance_passed':not failed,
            'float32_failed_conditions':failed,'float32_prediction_changes':prediction_changes,
            'float64_max_logit_error':maximum64,'float64_same_strict_tolerance_passed':True}

def baseline_controls(model,cache,cfg,out):
    path=out/'baseline_controls.json'
    if path.exists():
        import json
        assert json.loads(path.read_text())['passed'];return
    rows=[]
    # All controls complete before any memory-only treatment is computed.
    for di,d in enumerate(cfg['timing']['delays']):
        for suffix in cfg['suffixes']:
            m,c=cache['memory'][di],cache['control'][di]
            native=continue_pairs(model,m,c,range(18),range(18),suffix,cfg)
            natural=natural_outputs(model,cache,di,suffix,cfg)
            np.testing.assert_allclose(native,natural,atol=1e-5,rtol=1e-5)
            sham=continue_pairs(model,m.copy(),c.copy(),range(18),range(18),suffix,cfg)
            assert np.array_equal(native,sham)
            donors,receivers=np.indices((18,18));same=np.array([CONDITIONS[a][0]==CONDITIONS[b][0] for a,b in zip(donors.ravel(),receivers.ravel())])
            donors,receivers=donors.ravel()[same],receivers.ravel()[same]
            full=continue_pairs(model,m,c,donors,donors,suffix,cfg)
            np.testing.assert_allclose(full,natural[donors],atol=1e-5,rtol=1e-5)
            np.savez_compressed(out/f'baseline_d{d}_S{suffix}.npz',native_logp=native,natural_logp=natural,
                                sham_logp=sham,full_logp=full,full_donor=donors,full_receiver=receivers)
            rows.append({'d':d,'S':suffix,'native_cache_max_logp_error':float(np.abs(native-natural).max()),
                         'sham_max_logp_error':0.,'full_max_logp_error':float(np.abs(full-natural[donors]).max())})
    cpu_error=cpu_strict_control(model,cache,cfg)
    write_json(path,{'passed':True,'GPU_logp_atol':1e-5,'GPU_logp_rtol':1e-5,'CPU_logit_atol':2e-7,
                    'CPU_logit_rtol':2e-6,'CPU_uninterrupted':cpu_error,'strata':rows})
