
import numpy as np
from QuantumSparse.operator import Operator
from QuantumSparse.matrix import Matrix
from QuantumSparse.global_variables import NDArray
from typing import List, Union
from .tools import embed, check_orthogonality, Hilbert_Schmidt
from warnings import warn
#OpArr = NDArray[Operator]
OpArr = Union[List[Operator],np.ndarray]

class LocalHilbertSpace:
    dim:np.ndarray
    def __init__(self,dim:int) -> None:
        assert dim > 0, "dim must be positive"
        self.dim = dim
        
    def get_operator_basis(self,only_hermitian=False):
        """Returns the basis of the Hilbert space of the hermitian operators of each site"""
        if only_hermitian:
            warn("check that the number of operators is correct")
            dimOp = self.dim*2
            operators:List[Operator] = [None]*dimOp#np.zeros(dimOp,dtype=Operator)
            k = 0 
            for i in range(self.dim): # diagonal
                for j in range(self.dim):
                    # tmp:Operator = (Operator.one_hot(dim,i,j) + Operator.one_hot(dim,j,i))/2
                    # print(tmp.todense())
                    # OpBasis[n][k] = tmp
                    # k += 1
                    if i > j:
                        tmp:Operator = (1.j*Operator.one_hot(self.dim,i,j) -1.j* Operator.one_hot(self.dim,j,i))/np.sqrt(2)
                        print(tmp.todense())
                        operators[k] = tmp
                        k += 1
                    elif i < j:
                        tmp:Operator = (Operator.one_hot(self.dim,i,j) + Operator.one_hot(self.dim,j,i))/np.sqrt(2)
                        print(tmp.todense())
                        operators[k] = tmp
                        k += 1
                    else:
                        tmp:Operator = Operator.one_hot(self.dim,i,i)
                        print(tmp.todense())
                        operators[k] = tmp
                        k += 1
                assert np.all([Op.is_hermitean() for Op in operators]), "Operators are not hermitian"
        else:
            operators = [None]*(self.dim**2)
            k = 0 
            for i in range(self.dim):
                for j in range(self.dim): 
                    operators[k] = Operator.one_hot(self.dim,i,j)
                    k += 1

        assert check_orthogonality(operators,Hilbert_Schmidt), "Operators are not orthogonal"
        
        return np.asarray(operators,dtype=object)

class HilbertSpace:
    
    local_dims:np.ndarray    
    def __init__(self,local_dims:np.ndarray) -> None:
        assert local_dims.ndim == 1, "local_dims must be a 1D array"
        self.local_dims = local_dims
        
    def get_operator_basis(self,only_hermitian=False,embedded=True):
        """Returns the basis of the Hilbert space of the hermitian operators of each site"""
        N = len(self.local_dims)
        OpBasis = [None]*N# np.zeros(N,dtype=object) 
        for n,dim in enumerate(self.local_dims):    
            OpBasis[n] = LocalHilbertSpace(dim).get_operator_basis(only_hermitian=only_hermitian)
        OpBasis = np.asarray(OpBasis,dtype=object)
        if not embedded:
            return OpBasis
        else:
            OpBasis = embed(OpBasis)
            return OpBasis
        
        # Sx = np.zeros(NSpin,dtype=object) # S x
        # Sy = np.zeros(NSpin,dtype=object) # S y
        # Sp = np.zeros(NSpin,dtype=object) # S y
        # Sm = np.zeros(NSpin,dtype=object) # S y
        iden = Operator.identity(self.local_dims)
        for ii in range(len(operators)):
            for i in range(N):
                Ops = iden.copy()
                Ops[i] = operators[ii]
                OpBasis[n][ii][i] = Ops[0]
                for j in range(1,N):
                    OpBasis[n][ii][i] = Operator.kron(OpBasis[n][ii][i],Ops[j])
        
        for ii,zpm,out in enumerate(zip(operators,OpBasis[n])):
            for i in range(N):
                Ops = iden.copy()
                Ops[i] = zpm[i]
                out[i] = Ops[0]
                for j in range(1,N):
                    out[i] = Operator.kron(out[i],Ops[j]) 
                    
        # for i in range(NSpin):
        #     Sx[i] = compute_sx(Sp[i],Sm[i])
        #     Sy[i] = compute_sy(Sp[i],Sm[i])
            
        # return Sx,Sy,Sz,Sp,Sm       
            
        return OpBasis

def get_projectors_on_operator_basis(OpBasis:NDArray[OpArr]=None)->NDArray[OpArr]:
    N = len(OpBasis)
    OpProj = np.zeros(N,dtype=object)
    for n in range(N):
        dimOp = len(OpBasis[n])
        OpProj[n] = np.zeros(dimOp,dtype=object) 
        for i in range(dimOp):
            a = np.asarray(OpBasis[n][i].todense())
            OpProj[n][i] = OpBasis[n][i] #np.outer(np.conjugate(a.T),a)# projector(OpBasis[n][i])
    return OpProj

def get_projectors_on_site_operator(OpProj:NDArray[OpArr]=None)->OpArr:
    N = len(OpProj)
    Proj = np.zeros(N,dtype=object)
    for n in range(N):
        Proj[n] = OpProj[n].sum()/np.sqrt(len(OpProj[n]))
        assert np.allclose((Proj[n]@Proj[n]).todense(),Proj[n].todense()), "Projection is not idempotent"
    return Proj
