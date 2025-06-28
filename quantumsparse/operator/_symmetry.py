import numpy as np
from ._operator import Operator
from typing import TypeVar    
T = TypeVar('T',bound="Symmetry") 

class Symmetry(Operator):
    
    # def __init__(self:T,Op:Operator):
    #     if not Op.is_unitary():
    #         raise ValueError("Operator is not unitary")
    #     else:
    #         self = Op.copy()
    
    def set_eigen(self:T,w:np.ndarray,f:np.ndarray):
        if self.shape[0] != len(w):
            raise ValueError("worng number of eigenvalues")
        elif not np.allclose(np.abs(w),1):
            raise ValueError("not all eigenvalues have modulus 1")
        else:
            self.eigenvalues = w   
        
        assert self.shape == f.shape, "error"
        self.eigenstates = f
        self.normalize_eigevecs()
        return self.test_eigensolution()         
        
    # def levels2eigenstates(self:T,energy_levels):
    #     I = self.identity(self.shape[0])
    #     for w in energy_levels:
    #         M = Operator(self - I * w)
    #         f = M.kernel()
    #     pass
    
import cmath

