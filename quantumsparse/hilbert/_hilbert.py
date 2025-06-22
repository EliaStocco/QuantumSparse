
import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.matrix import Matrix, State
from quantumsparse.global_variables import NDArray
from typing import List, Union, Optional
from .tools import embed_operators, embed_states, check_orthogonality, Hilbert_Schmidt
from warnings import warn
from functools import wraps
from typing import Optional, Callable
# import xarray as xr
#OpArr = NDArray[Operator]
OpArr = Union[List[Operator],np.ndarray]

DEBUG = True

class LocalHilbertSpace:
    dim:np.ndarray
    def __init__(self,dim:int) -> None:
        assert dim > 0, "dim must be positive"
        self.dim = dim
        
    def raising(self)->Operator:
        """
        Returns a raising operator in the local Hilbert space.        
        
        Parameters
        ----------
        ket : int
            The ket index of the raising operator.
        test : bool, optional
            Whether to test if the raising operator is hermitian (default is DEBUG).
        
        Returns
        -------
        Operator
            The raising operator in the local Hilbert space.
        """
        op:Operator = Operator((self.dim,self.dim))
        for n in range(self.dim-1):
            op[n,n+1] = 1
        return op
    
    def lowering(self)->Operator:
        """
        Returns a lowering operator in the local Hilbert space.
        
        The returned operator is:
        
        L = |ket><ket|
        
        Parameters
        ----------
        ket : int
            The ket index of the lowering operator.
        test : bool, optional
            Whether to test if the lowering operator is hermitian (default is DEBUG).
        
        Returns
        -------
        Operator
            The lowering operator in the local Hilbert space.
        """
        op:Operator = Operator((self.dim,self.dim))
        for n in range(self.dim-1):
            op[n+1,n] = 1
        return op
        
    def projector(self,ket:int,bra:int,test:Optional[bool]=DEBUG)->Operator:
        """
        Returns a projector operator in the local Hilbert space.
        
        The returned operator is:
        
        P = |ket><bra| 
        
        Parameters
        ----------
        ket : int
            The ket index of the projector.
        bra : int
            The bra index of the projector.
        test : bool, optional
            Whether to test if the projector is hermitian (default is DEBUG).
        
        Returns
        -------
        Operator
            The projector operator in the local Hilbert space.
        """
        op:Operator = Operator.one_hot(self.dim,ket,bra)
        assert op.shape == (self.dim, self.dim), "operator shape is not correct"
        assert op[ket,bra] == 1, "operator is not one-hot"
        if test:
            opdagger = self.projector(bra,ket,test=False)
            diff:Operator = opdagger - op.dagger()
            assert diff.norm() < 1e-12, "projector is not hermitian"
        return op
        
    def get_basis(self)->List[State]:
        states = [None]*self.dim
        for n in range(self.dim):
            states[n] = State.one_hot(self.dim,n)
        return states
        
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
        
        return operators
        # out = 
        # return np.asarray(operators,dtype=object)
def embed(normalize: bool = True) -> Callable:
    """Decorator factory to perform hermitian test and embed an operator with normalization option."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, site: int, **argv) -> Operator:
            # Step 1: Generate the initial operator using the original function
            op:Operator = func(self, site, **argv)        
            # Step 3: Embed operator
            ops = [None] * len(self.local_dims)
            for n in range(len(self.local_dims)):
                ops[n] = []
            ops[site] = [op]
            ops = embed_operators(ops, self.local_dims, normalize=normalize)
            
            # Step 4: Return embedded operator
            return ops[site][0]
        
        return wrapper
    return decorator

class HilbertSpace:
    
    local_dims:np.ndarray   
    local_Hilbert_space: List[LocalHilbertSpace]
     
    def __init__(self,local_dims:np.ndarray) -> None:
        assert local_dims.ndim == 1, "local_dims must be a 1D array"
        self.local_dims = local_dims
        self.local_Hilbert_spaces = [LocalHilbertSpace(dim) for dim in local_dims]
        

    @embed(normalize=False)
    def raising(self,site:int)->Operator:
        """
        Returns a raising operator in the Hilbert space of the whole system.
        
        The returned operator is:
        
        S+ = Sx + i Sy
        
        Parameters
        ----------
        ket : int
            The ket index of the raising operator.
        bra : int
            The bra index of the raising operator.
        site : int
            The site index where the raising operator is applied.
        test : bool, optional
            Whether to test if the raising operator is hermitian (default is DEBUG).
        
        Returns
        -------
        Operator
            The raising operator in the Hilbert space of the whole system.
        """
        return self.local_Hilbert_spaces[site].raising() 
    
    @embed(normalize=False)
    def lowering(self,site:int)->Operator:
        """
        Returns a lowering operator in the Hilbert space of the whole system.
        
        The returned operator is:
        
        S- = Sx - i Sy
        
        Parameters
        ----------
        ket : int
            The ket index of the lowering operator.
        bra : int
            The bra index of the lowering operator.
        site : int
            The site index where the lowering operator is applied.
        test : bool, optional
            Whether to test if the lowering operator is hermitian (default is DEBUG).
        
        Returns
        -------
        Operator
            The lowering operator in the Hilbert space of the whole system.
        """
        return self.local_Hilbert_spaces[site].lowering()
        
    @embed
    def projector(self,ket:int,bra:int,site:int,test:Optional[bool]=DEBUG)->Operator:
        """
        Returns a projector operator in the Hilbert space of the whole system.
        
        The returned operator is:
        
        P = |ket,site><bra,site| 
        
        Parameters
        ----------
        ket : int
            The ket index of the projector.
        bra : int
            The bra index of the projector.
        site : int
            The site index where the projector is applied.
        test : bool, optional
            Whether to test if the projector is hermitian (default is DEBUG).
        
        Returns
        -------
        Operator
            The projector operator in the Hilbert space of the whole system.
        """
        return self.local_Hilbert_spaces[site].projector(ket,bra)
    
        # ops = [None]*len(self.local_dims)
        # for n in range(len(self.local_dims)):
        #     ops[n] = []
        # ops[site] = [op]
        # ops = embed_operators(ops,self.local_dims,normalize=True)
        # op = ops[site][0]
        # return op        
        
    def get_basis(self,embedded=True):
        N = len(self.local_dims)
        basis = [None]*N
        for n,dim in enumerate(self.local_dims):    
            basis[n] = self.local_Hilbert_spaces[n].get_basis()
        if not embedded:
            return basis
        else:
            basis = embed_states(basis,self.local_dims)
            return basis
        
    def get_operator_basis(self,only_hermitian=False,embedded=True)->List[List[Matrix]]:
        """Returns the basis of the Hilbert space of the hermitian operators of each site"""
        N = len(self.local_dims)
        OpBasis = [None]*N# np.zeros(N,dtype=object) 
        for n,dim in enumerate(self.local_dims):    
            OpBasis[n] = self.local_Hilbert_spaces[n].get_operator_basis(only_hermitian=only_hermitian)
        if not embedded:
            return OpBasis
        else:
            EmbOpBasis = embed_operators(OpBasis,self.local_dims)
            # results = [mutual_product(ops,Hilbert_Schmidt) for ops in OpBasis] 
            # Embresults = [mutual_product(ops,Hilbert_Schmidt) for ops in EmbOpBasis] 
            return EmbOpBasis

    def get_projectors(self)->List[Matrix]:
        OpBasis = self.get_operator_basis(only_hermitian=False,embedded=True)
        projs = [None]*len(self.local_dims)
        for s in range(len(self.local_dims)): # cycle over Local Hilbert Spaces
            for n,op in enumerate(OpBasis[s]):
                if n == 0 :
                    projs[s] = op @ op.dagger()
                else:
                    projs[s] += op @ op.dagger()
            norm = np.sqrt(Hilbert_Schmidt(projs[s],projs[s]))
            projs[s] /= norm
            assert projs[s] @ projs[s] - projs[s], "Projectors are not idempotent."
        return projs
    

# def get_projectors_on_operator_basis(OpBasis:NDArray[OpArr]=None)->NDArray[OpArr]:
#     N = len(OpBasis)
#     OpProj = np.zeros(N,dtype=object)
#     for n in range(N):
#         dimOp = len(OpBasis[n])
#         OpProj[n] = np.zeros(dimOp,dtype=object) 
#         for i in range(dimOp):
#             a = np.asarray(OpBasis[n][i].todense())
#             OpProj[n][i] = OpBasis[n][i] #np.outer(np.conjugate(a.T),a)# projector(OpBasis[n][i])
#     return OpProj

# def get_projectors_on_site_operator(OpProj:NDArray[OpArr]=None)->OpArr:
#     N = len(OpProj)
#     Proj = np.zeros(N,dtype=object)
#     for n in range(N):
#         Proj[n] = OpProj[n].sum()/np.sqrt(len(OpProj[n]))
#         assert np.allclose((Proj[n]@Proj[n]).todense(),Proj[n].todense()), "Projection is not idempotent"
#     return Proj
