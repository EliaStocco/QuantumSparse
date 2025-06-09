import numpy as np
from QuantumSparse.operator import Operator
from QuantumSparse.matrix import Matrix, State
from QuantumSparse.global_variables import NDArray
from typing import List, Union, Optional
from copy import deepcopy
OpArr = Union[List[Operator],np.ndarray]

def projector(Op:Operator)->Operator:
    return Op.T@Op

def inner_product(A,B):
    return A@B

def Hilbert_Schmidt(A:Matrix,B:Matrix)->float:
    """
    Computes the Hilbert-Schmidt inner product of two matrices A and B.
    Attentioni: the norm of an operator is the square root of its Hilbert-Schmidt inner product.

    Parameters:
        A (Matrix): The first matrix.
        B (Matrix): The second matrix.

    Returns:
        float: The Hilbert-Schmidt inner product of A and B.
    """
    return (A.dagger() @ B).trace()

def mutual_product(Arr:np.ndarray,product=inner_product):
    N = len(Arr)
    results = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            results[i,j] = product(Arr[i],Arr[j])
    return results
    
def check_orthogonality(Arr:np.ndarray,product=inner_product):
    N = len(Arr)
    results = mutual_product(Arr,product)
    assert np.allclose(results,results.T), "The provided scalar product is not symmetric!"
    if not np.allclose(results,np.eye(N)):
        return False
    return True

def _embed_operators_(Op:Matrix,index:int,dims:List[int])->Matrix:
    iden:List[Operator] = Operator.identity(dims)
    M = len(dims)
    out = iden[0] if index > 0 else Op
    for n in range(1,M):
        if n == index:
            out = Operator.kron(out,Op)
        else:
            out = Operator.kron(out,iden[n])
    return out

def embed_states(states:List[List[State]], dims:List[int],normalize:Optional[bool]=True)->List[List[State]]:
    # dims = [ op[0].shape[0] for op in operators]
    States = [None]*len(states) # the lenght is the number of sites
    for n,S in enumerate(states): # cycle over sites
        States[n] = [ _embed_operators_(op,n,dims) for i,op in enumerate(S)]
        if normalize:
            for i,s in enumerate(States[n]):
                States[n][i] /= np.sqrt(inner_product(s,s)) # normalize the operators
            assert check_orthogonality(States[n]), "States are not orthogonal"
    return States 
    
def embed_operators(operators:List[List[Matrix]],dims:List[int],normalize:Optional[bool]=True)->List[List[Matrix]]:
    """
    Embeds a list of operators into the Hilbert space of the whole system.

    Args:
        operators (List[List[Matrix]]): A list of operators to be embedded.
        dims: The dimensions of the Hilbert space.

    Returns:
        List[List[Matrix]]: The embedded operators.
        
    Example:
        spin_values = [0.5,0.5,0.5]
        dims = [2,2,2]
        sz,sp,sm = single_Szpm(spin_values)
        Sz = embed_operators([[sz[0],sz[1],sz[2]]],dims)
        
        Sz,Sp,Sm = embed_operators([sz,sp,sm],dims)
    """
    # dims = [ op[0].shape[0] for op in operators]
    Operators = [None]*len(operators) # the lenght is the number of sites
    for n,ops in enumerate(operators): # cycle over sites
        Operators[n] = [ _embed_operators_(op,n,dims) for i,op in enumerate(ops)]
        if normalize:
            for i,op in enumerate(Operators[n]):
                Operators[n][i] /= np.sqrt(Hilbert_Schmidt(op,op)) # normalize the operators
            assert check_orthogonality(Operators[n],Hilbert_Schmidt), "Operators are not orthogonal"
    return Operators
        
    
    # iden:List[Operator] = Operator.identity(dims)
    # N = len(operators)
    # embedded_operators = [None]*N
    # for site in range(N):
    #     Ops = deepcopy(iden)
    #     M = len(operators[site])
    #     embedded_operators[site] = [None]*M
    #     for i in range(M):
            
    #         Ops[i] = operators[ii]
    #         OpBasis[n][ii][i] = Ops[0]
    #         for j in range(1,N):
    #             OpBasis[n][ii][i] = Operator.kron(OpBasis[n][ii][i],Ops[j])
                
                
    # for i in range(NSpin):
    #     Ops = iden.copy()
    #     Ops[i] = zpm[i]
    #     out[i] = Ops[0]
    #     for j in range(1,NSpin):
    #         out[i] = Operator.kron(out[i],Ops[j]) 
    
    # for ii,zpm,out in enumerate(zip(operators,OpBasis[n])):
    #     for i in range(N):
    #         Ops = iden.copy()
    #         Ops[i] = zpm[i]
    #         out[i] = Ops[0]
    #         for j in range(1,N):
    #             out[i] = Operator.kron(out[i],Ops[j]) 