
import pytest 
import cmath
import numpy as np
from typing import List
from quantumsparse.matrix import Matrix

#----------------------------------#
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

#-----------------#
@pytest.mark.parametrize("N", [1, 2, 4, 5])
def test_roots_of_unity(N):
    r = roots_of_unity(N)
    assert len(r) == N, "Length mismatch"

    # Check all roots satisfy z^N == 1
    for root in r:
        val = root ** N
        assert abs(val - 1) < 1e-12, f"Root {root} to the power {N} != 1"

    # Check roots are distinct
    assert len(np.unique(np.round(r, decimals=12))) == N, "Roots are not distinct"
    
#----------------------------------#
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
    
    result = Ops[0].copy()
    for op in Ops[1:]:
        result = result @ op
    
    return result

#----------------------------------#
def unique_with_tolerance(arr, tol=1e-8):
    """
    Returns the unique elements of an array within a specified tolerance.

    Parameters:
    arr (array_like): The input array.
    tol (float, optional): The tolerance for uniqueness. Defaults to 1e-8.

    Returns:
    tuple: A tuple containing the unique elements and their indices.
    """
    rounded_arr = np.round(arr, decimals=int(-np.log10(tol)))
    unique_rounded,index = np.unique(rounded_arr,return_inverse=True)
    return unique_rounded, index


        