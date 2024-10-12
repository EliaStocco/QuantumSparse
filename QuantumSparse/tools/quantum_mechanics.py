import numpy as np
from QuantumSparse.tools.optimize import jit
from QuantumSparse.operator import Operator
from QuantumSparse.matrix import Matrix

#@jit
def expectation_value(Op:Operator,Psi:Operator)->np.ndarray:
    braket = Psi.dagger() @ Op @ Psi
    return braket.real
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

def projector(Op:Operator)->Operator:
    return Op.T@Op

def inner_product(A,B):
    return A@B

def Hilbert_Schmidt(A:Matrix,B:Matrix)->float:
    return (A.dagger() @ B).trace()

def check_orthogonality(Arr:np.ndarray,product=inner_product):
    N = len(Arr)
    results = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            results[i,j] = product(Arr[i],Arr[j])
    assert np.allclose(results,results.T), "The provided scalar product is not symmetric!"
    if not np.allclose(results,np.eye(N)):
        return False
    return True

