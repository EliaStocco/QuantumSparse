
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

# #----------------------------------#
# def align_eigenvectors(eigvec1: np.ndarray, eigvec2: np.ndarray, atol: float = 1e-10):
#     """
#     Align eigvec1 to eigvec2 using Procrustes analysis.
#     Returns the optimal orthogonal matrix U such that eigvec1 @ U ~ eigvec2
#     """
#     # eigvec1, eigvec2 should be 2D (columns = eigenvectors)
#     assert eigvec1.shape == eigvec2.shape
    
#     from scipy.linalg import orthogonal_procrustes

#     # Compute the orthogonal Procrustes solution
#     U, scale = orthogonal_procrustes(eigvec1, eigvec2)
    
#     # Apply U to eigvec1
#     aligned = eigvec1 @ U
    
#     # Compute difference
#     diff = np.linalg.norm(aligned - eigvec2) / eigvec1.shape[0]
#     assert diff < atol, f"Eigenvectors should match after alignment (diff={diff})"
    
#     return U, aligned, diff

def align_eigenvectors(vecs1: np.ndarray, vecs2: np.ndarray):
    """
    Find optimal unitary/orthogonal U so that vecs1 @ U ~ vecs2
    Works for degenerate eigenvectors.
    """
    from scipy.linalg import svd
    M = vecs1.conj().T @ vecs2
    U, _, Vh = svd(M)
    return vecs1 @ (U @ Vh)

def is_unitary(U, atol=1e-10):
    """
    Check if a dense matrix U is unitary.
    
    Parameters
    ----------
    U : np.ndarray
        Square matrix to check
    atol : float
        Absolute tolerance for numerical comparison
    
    Returns
    -------
    bool
        True if U is unitary within tolerance
    """
    U = np.asarray(U)
    assert U.shape[0] == U.shape[1], "Matrix must be square"
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=atol)