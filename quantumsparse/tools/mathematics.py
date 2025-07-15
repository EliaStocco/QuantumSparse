import cmath
import numpy as np
from typing import List
from quantumsparse.matrix import Matrix

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

def product(Ops:List[Matrix])->Matrix:
    """
    Calculate the product of a list of operators.
    
    Parameters:
        Ops (List[Matrix]): A list of operators to be multiplied.
    
    Returns:
        Matrix: The product of the operators.
    """
    if not Ops:
        raise ValueError("The list of operators is empty.")
    
    result = Ops[0]
    for op in Ops[1:]:
        result = result @ op
    
    return result