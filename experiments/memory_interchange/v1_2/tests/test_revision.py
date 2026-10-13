import itertools
import numpy as np
from ...truth import apply as apply_word
from ..diagnostics import candidate_maps, metadata, rates, CANDIDATES
from ..common import config
from ...v1_1.audit import gate

def test_candidate_oracle_order_and_unidentifiable_reversal():
    for p in itertools.islice(itertools.permutations(range(7)),0,5040,23):
        p=np.array(p)
        for ops in ('II','CC','IC','CI'):
            m,inv=candidate_maps(p,ops)
            word=ops.replace('I','A').replace('C','B')
            assert np.array_equal(m[0],apply_word(p,word,0))
            assert np.array_equal(m[4],apply_word(p,word[::-1],0))
            assert np.array_equal(np.take_along_axis(m,inv,1),np.tile(np.arange(7),(5,1)))
            if ops in ('II','CC'):assert np.array_equal(m[0],m[4])
    p=np.array([1,3,0,5,2,6,4]);m,_=candidate_maps(p,'IC')
    assert not np.array_equal(m[0],m[4])

def test_visible_semantics_and_candidate_coincidences_are_kept():
    d={'words':np.array(['NABN','NBAN','NBAN','NABN']),
       'context':np.array([0,1,0,1]),'delay':np.array([0,10,0,10]),'kind':np.array([0,1,0,1])}
    info=metadata(d)
    assert info['real_operation'].tolist()==['IC','IC','CI','CI']
    assert info['visible_order'].tolist()==['AB','BA','BA','AB']
    assert set(info['slots'])=={'2,3'}
    answers=np.array([[0,0,1,2,0],[0,1,1,2,2]])
    result=rates(np.array([0,2]),answers)
    assert result['reversed']['all']['matches']==2
    assert result['reversed']['differs_from_correct']['n']==1
    assert result['reversed']['unique_among_candidates']['n']==0

def test_budgets_and_original_factors_unchanged():
    c=config()
    assert sum(s['max_updates'] for s in c['stages'])==200000
    assert [s['max_updates'] for s in c['stages']]==[20000]*5+[50000]*2
    assert c['validation']['overall_target']==.97 and c['validation']['factor_target']==.95
    assert c['validation']['consecutive_passes']==2
    assert c['formal_seeds']==list(range(101,107))
    x={'accuracy':.99,'strata':{name:{'x':{'accuracy':.96}} for name in c['validation']['gate_factors']}}
    assert gate(x,c)
    x['strata']['true_operation_sequence']['x']['accuracy']=.949
    assert not gate(x,c)
