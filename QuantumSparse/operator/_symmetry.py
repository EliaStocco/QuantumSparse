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
    
    def set_eigenvalues(self:T,w):
        if self.shape[0] != len(w):
            raise ValueError("worng number of eigenvalues")
        elif not np.allclose(np.abs(w),1):
            raise ValueError("not all eigenvalues have modulus 1")
        else:
            self.eigenvalues = w            
        
    def levels2eigenstates(self:T,energy_levels):
        I = self.identity(self.shape[0])
        for w in energy_levels:
            M = Operator(self - I * w)
            f = M.kernel()
        pass
    
import cmath

def roots_of_unity(N:int)->np.ndarray:
    """
    Calculate the N-th roots of unity.
    
    Parameters:
        N (int): The degree of the roots of unity.
    
    Returns:
        list: A list of the N-th roots of unity as complex numbers.
    """
    roots = np.zeros(N, dtype=complex)
    for k in range(N):
        # Calculate the k-th root of unity
        angle = 2 * cmath.pi * k / N
        root = cmath.exp(1j * angle)
        roots[k] = root
    
    return roots