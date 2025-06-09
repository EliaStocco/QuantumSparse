from quantumsparse.operator import Operator
from quantumsparse.hilbert import HilbertSpace
from quantumsparse.hilbert.tools import mutual_product, Hilbert_Schmidt, check_orthogonality
import numpy as np
from icecream import ic
from typing import TypeVar, Type, List
T = TypeVar('T', bound='Translation')

class Translation(Operator):
    """
    This class is a subclass of quantumsparse.operator.Operator and is used to represent the translation operator.
    """
    
    @classmethod
    def from_dims(cls: Type[T], N: int, M: int) -> T:
        translation = create_translation_operator(int(N),int(M))
        return cls(translation)
    
def create_translation_operator(N: int, M: int) -> Operator:
    assert N > 0 , "number of sites must be positive"
    assert M > 0 , "local Hilbert space dimension must be positive"
    
    local_dims = np.full(N,M)
    HS = HilbertSpace(local_dims)
                    
    T = None
    for n in range(N):
        n_plus = n+1 if n < N-1 else 0
        R = HS.lowering(n)
        L = HS.raising(n_plus)
        T = R@L + L@R if T is None else T + R@L + L@R
        for _ in range(M):
            R @= R
            L @= L
            tmp = R@L + L@R
            T += tmp
    return T
                
    
    
# def create_translation_operator(N: int, M: int) -> Operator:
#     assert N > 0 , "number of sites must be positive"
#     assert M > 0 , "local Hilbert space dimension must be positive"
    
#     local_dims = np.full(N,M)
#     HS = HilbertSpace(local_dims)
#     # tmp = list()
#     # for n in range(N):
#     #     for m in range(M):
#     #         tmp.append(HS.projector(0,m,n))
#     # test = mutual_product(tmp,Hilbert_Schmidt)
    
#     # T = None
#     # for n in range(N):
#     #     n_plus = n+1 if n < N-1 else 0
#     #     for m1 in range(M):
#     #         destruct = None
#     #         construct = None
#     #         for m2 in range(M):
#     #             destruct  = HS.projector(m2,m1,n)               if destruct  is None else destruct  + HS.projector(m2,m1,n)
#     #             construct = HS.projector(m2,m1,n_plus).dagger() if construct is None else construct + HS.projector(m2,m1,n_plus).dagger()
#     #             T = construct @ destruct if T is None else T + construct @ destruct
#     #             # # assert (construct - HS.projector(m,0,n_plus)).norm() < 1e-12, "construct - HS.projector(m,0,n_plus) != 0"
#     #             # update = construct @ destruct
#     #             # # ic(update.todense())
#     #             # if T is None:
#     #             #     T = update
#     #             # else:
#     #             #     T += update
                    
                    
#     T = None
#     for n in range(N):
#         n_plus = n+1 if n < N-1 else 0
#         Tn = None
#         for m1 in range(M):
#             for m2 in range(M):
#                 # if m2 != 0 :
#                 #     continue
#                 ic(n,n_plus,m1,m2)
#                 destruct = HS.projector(m2,m1,n)
#                 construct = HS.projector(m2,m1,n_plus).dagger()
#                 # assert (construct - HS.projector(m,0,n_plus)).norm() < 1e-12, "construct - HS.projector(m,0,n_plus) != 0"
#                 update = construct @ destruct
#                 # ic(update.todense())
#                 T = update if T is None else T + update
#         # # ic(Tn.todense())
#         # if T is None:
#         #     T = Tn
#         # else:
#         #     T += Tn
#     return T
                
#     test = HS.projector(0,0,0)
#     test = HS.projector(1,0,0)
#     test = HS.projector(1,1,0)
#     test = HS.projector(0,0,1)
#     # basis = HS.get_basis()
#     return
    