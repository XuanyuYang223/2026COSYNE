"""Synthetic implementation checks, independent of the observed consistency."""
import numpy as np
from .common import probabilities,FLAG
from .analyze import compare
from .summarize import measure

def run():
    # A pair of identical illegal prediction vectors must count as agreement,
    # not be filtered or mislabeled as a valid/correct permutation.
    shape=(1,1,2,3,2,7)
    pa=np.zeros(shape,np.uint8);da=pa.copy();truth=np.broadcast_to(np.arange(7),shape)
    p=np.broadcast_to(np.array([.4,.4,.04,.04,.04,.04,.04]),(*shape,7)).copy()
    av=np.ones(shape,bool);ties=av.copy()
    flags,diff,answers=compare(p,p,pa,da,truth,av,av,ties,ties,av,av)
    m=measure(flags,diff,answers,'all')
    assert m['answer_agreement_pooled']==1 and m['full14_agreement']==1
    assert m['patch_nonbijective_or_noninverse_vector_n']==6
    assert m['patch_tie_n']==84
    error=measure(flags,diff,answers,'error')
    assert error['n_subgroup_queries']==72 and error['answer_agreement_pooled']==1
    pa[0,0,0,0,1,6]=1
    flags,diff,answers=compare(p,p,pa,da,truth,av,av,ties,ties,av,av)
    m=measure(flags,diff,answers,'all')
    assert m['answer_disagree_n']==1 and m['full14_agree_n']==5 and m['item7_agree_n']==6 and m['position7_agree_n']==5
    lp=np.log(np.array([.4,.4,.04,.04,.04,.04,.04]));pp,valid,diagnostic=probabilities(lp)
    assert valid and pp.argmax()==0 and abs(pp.sum()-1)<1e-15
    empty=measure(flags[:,:,0:0],diff[:,:,0:0],answers[:,:,0:0],'all')
    assert empty['row_status']=='EMPTY' and empty['n_subgroup_queries']==0
    print('Synthetic checks passed: donor wrong-output reproduction, tied argmax, illegal vectors, 14/7 consistency, empty strata and float64 probabilities.')

if __name__=='__main__':run()
