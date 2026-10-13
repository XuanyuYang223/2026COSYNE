import numpy as np
from ...common import config as original_config
from ...truth import CONDITIONS,matrix_apply,preimage
from ...geometry import raw_distance
from ...v1_1.audit import gate
from ..common import config
from ..intervene import frozen_masks,initial_counterfactual,quality
from ..predict import bootstrap_labels,folds

def test_masks_preserve_donor_own_correctness_and_wrong_treatment_mask():
    nc=np.zeros((18,4,2,7),bool);nc[0,0,0,0]=True;nc[1,0,0,0]=True
    nc[0,1,0,0]=True;mismatch=np.array([1,0,3,2]);m=frozen_masks(nc,mismatch)
    assert m['rec'][2,1,0,0,0] and not m['both'][2,1,0,0,0]
    assert m['both'][0,1,0,0,0] and m['three'][0,1,0,0,0]
    assert m['rec'][0,1,0,0,0]==nc[1,0,0,0]
    assert quality(np.array([.899,.90,.949,.95])).tolist()==[0,1,1,2]

def test_initial_counterfactual_uses_recipient_prefix_then_suffix():
    q=np.array([[1,3,0,5,2,6,4],[0,2,4,6,1,3,5]])
    initial=np.array([[preimage(q,h,r) for r,h in CONDITIONS]])
    cf=initial_counterfactual({'initial':initial},0,'NA')
    for a,(ra,ha) in enumerate(CONDITIONS):
        for b,(rb,hb) in enumerate(CONDITIONS):
            assert np.array_equal(cf[a,b],matrix_apply(initial[0,a],hb+'NA',rb))
        assert np.array_equal(cf[a,a],matrix_apply(q,'NA',ra))
    x=np.eye(7)[q].reshape(2,49)
    assert raw_distance(x,x)==0
    assert np.isclose(raw_distance(x,np.eye(7)[6-q].reshape(2,49)),12/7)

def test_only_stage7_fine_gate_is_removed():
    c=config();m={'accuracy':.98,'strata':{k:{'x':{'accuracy':.96}} for k in c['validation']['gate_factors']}}
    m['strata']['true_operation_sequence']['x']['accuracy']=.91
    assert not gate(m,c)
    coarse=dict(c,validation={**c['validation'],'gate_factors':c['validation']['stage7_gate_factors']})
    assert gate(m,coarse)
    assert sum(s['max_updates'] for s in c['stages'])==200000

def test_block_bootstrap_uses_mask_denominators_and_both_directions():
    b={'loss_sums':np.zeros((1,18,18,2,4)), 'mask_counts':np.zeros((1,18,18,2,4)),
       'delays':np.array([0]),'suffixes':np.array(['NN'])}
    b['loss_sums'][0,0,1,:,1]=[2,12];b['loss_sums'][0,1,0,:,1]=[4,8]
    b['mask_counts'][0,0,1,:,1]=[1,2];b['mask_counts'][0,1,0,:,1]=[2,3]
    r=[{'a':0,'b':1,'d':0,'S':'NN'}]
    assert np.isclose(bootstrap_labels(r,b,np.array([1,0]),1)[0],2)
    assert np.isclose(bootstrap_labels(r,b,np.array([1,1]),1)[0],26/8)
