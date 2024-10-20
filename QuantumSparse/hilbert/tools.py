import numpy as np
from QuantumSparse.operator import Operator
from QuantumSparse.matrix import Matrix
from QuantumSparse.global_variables import NDArray
from typing import List, Union
from copy import deepcopy
OpArr = Union[List[Operator],np.ndarray]

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

def embed(operators):
    dims = [ op[0].shape[0] for op in operators]
    iden:List[Operator] = Operator.identity(dims)
    N = len(operators)
    embedded_operators = [None]*N
    for site in range(N):
        Ops = deepcopy(iden)
        M = len(operators[site])
        embedded_operators[site] = [None]*M
        for i in range(M):
            
            Ops[i] = operators[ii]
            OpBasis[n][ii][i] = Ops[0]
            for j in range(1,N):
                OpBasis[n][ii][i] = Operator.kron(OpBasis[n][ii][i],Ops[j])
                
                
    for i in range(NSpin):
        Ops = iden.copy()
        Ops[i] = zpm[i]
        out[i] = Ops[0]
        for j in range(1,NSpin):
            out[i] = Operator.kron(out[i],Ops[j]) 
    
    for ii,zpm,out in enumerate(zip(operators,OpBasis[n])):
        for i in range(N):
            Ops = iden.copy()
            Ops[i] = zpm[i]
            out[i] = Ops[0]
            for j in range(1,N):
                out[i] = Operator.kron(out[i],Ops[j]) 