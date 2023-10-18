# quantum mechanis
from scipy import sparse
import numpy as np

# to be modified
def expectation_value(Op,Psi):
    V  = sparse.csr_matrix(Psi)
    Vc = V.conjugate(True)
    return ((Op @ V).multiply(Vc)).toarray().real.sum(axis=0)

def standard_deviation(Op,Psi,mean=None):
    if mean is None :
        mean = expectation_value(Op,Psi)
    Quad = expectation_value(Op@Op,Psi)
    return np.sqrt( Quad - mean**2)