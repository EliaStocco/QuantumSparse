import numpy as np
from quantumsparse.tools.optimize import jit
from quantumsparse.operator import Operator
from quantumsparse.matrix import Matrix

#@jit
def expectation_value(Op:Operator,Psi:Operator)->np.ndarray:
    OPsi = Op @ Psi
    tmp = OPsi.multiply(Psi.dagger()).real
    return np.asarray(tmp.sum(axis=0)).flatten()
    # braket = Psi.dagger() @ Op @ Psi
    # return braket.real
    # V  = sparse.csr_matrix(Psi)
    # Vc = V.conjugate(True)
    # return ((Op @ V).multiply(Vc)).toarray().real.sum(axis=0)

# # to be modified
# def expectation_value(Op,Psi):
#     V  = sparse.csr_matrix(Psi)
#     Vc = V.conjugate(True)
#     return ((Op @ V).multiply(Vc)).toarray().real.sum(axis=0)

#@jit
def standard_deviation(Op:Operator,Psi:Operator,mean:np.ndarray=None)->np.ndarray:
    if mean is None :
        mean = expectation_value(Op,Psi)
    Quad = expectation_value(Op@Op,Psi)
    return np.sqrt( Quad - mean**2)



