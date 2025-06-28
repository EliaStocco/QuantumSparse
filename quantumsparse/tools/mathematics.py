import cmath
import numpy as np

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