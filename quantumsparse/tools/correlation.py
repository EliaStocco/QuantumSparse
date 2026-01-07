# some function recalling statistical physics results
from .quantum_mechanics import expectation_value

def correlation(OpA,OpB,Psi):

    meanA =  expectation_value(OpA,Psi)
    meanB =  expectation_value(OpB,Psi)
    square = expectation_value(OpA@OpB,Psi)
    
    return square - meanA*meanB


