import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.tools.mathematics import align_eigenvectors, is_unitary

def compare_eigensolutions(H1:Operator, H2:Operator, atol:float=1e-10)->None:
    
    # ------------------------- # 
    # sanity check
    assert H1.shape == H2.shape, "Operators must have the same shape"
    assert H1.is_diagonalized(), "H1 is not diagonalized"
    assert H2.is_diagonalized(), "H2 is not diagonalized"
    
    # ------------------------- # 
    # check consistency
    N,M = H1.shape
    N = N*M
    
    test = H1.test_eigensolution().norm() / N 
    assert test < atol, "Eigensolution of H1 is not correct"
    
    test = H2.test_eigensolution().norm() / N 
    assert test < atol, "Eigensolution of H2 is not correct"
    
    
    # ------------------------- # 
    # eigenvalues
    _H1 = H1.sort()
    _H2 = H2.sort()
    assert np.allclose(_H1.eigenvalues, _H2.eigenvalues, atol=atol), \
        "Eigenvalues should be identical"
        
    # ------------------------- # 
    # eigenvectors
    
    # eigenstates can differ by a unitary transform (degenerate subspaces)
    diff = (_H1.eigenstates - _H2.eigenstates).norm() / N
    if diff > atol:
        U:Operator = _H2.eigenstates @ _H1.eigenstates.dagger()
        assert U.is_unitary(), "U should be a unitary transformation" # of course by construction
        
        diff = (U@_H1.eigenstates - _H2.eigenstates).norm() / N
        assert diff < atol, "Eigenstates should match up to a unitary transformation"
        
        _H2.eigenstates = U@_H1.eigenstates
        test = _H2.test_eigensolution().norm() / N 
        assert test < atol, "Eigensolution of _H1 is not correct"
        
def compare_eigensolutions_dense_real(
    eigval1: np.ndarray,
    eigval2: np.ndarray,
    eigvec1: np.ndarray,
    eigvec2: np.ndarray,
    atol: float = 1e-10,
) -> None:
    """
    Compare two eigensolutions (eigenvalues and eigenvectors) for real dense matrices.

    Parameters
    ----------
    eigval1, eigval2 : np.ndarray
        Eigenvalue arrays (1D).
    eigvec1, eigvec2 : np.ndarray
        Eigenvector arrays (2D, column-wise).
    atol : float
        Absolute tolerance for comparison.
    """
    # sanity checks
    assert eigval1.shape == eigval2.shape, "Eigenvalue arrays must have same shape"
    assert eigvec1.shape == eigvec2.shape, "Eigenvector arrays must have same shape"
    assert np.allclose(eigval1.imag, 0, atol=atol), "eigval1 must be real"
    assert np.allclose(eigval2.imag, 0, atol=atol), "eigval2 must be real"

    # sort eigenpairs by eigenvalues
    idx1 = np.argsort(eigval1.real)
    idx2 = np.argsort(eigval2.real)
    eigval1_sorted = eigval1[idx1].real
    eigval2_sorted = eigval2[idx2].real
    eigvec1_sorted = eigvec1[:, idx1]
    eigvec2_sorted = eigvec2[:, idx2]

    # compare eigenvalues
    assert np.allclose(eigval1_sorted, eigval2_sorted, atol=atol), \
        "Eigenvalues should be identical"

    # compare eigenvectors (up to a unitary / orthogonal transform)
    N = eigvec1.shape[0]
    diff = np.linalg.norm(eigvec1_sorted - eigvec2_sorted) / N
    if diff > atol:
        # check if they differ by an orthogonal matrix
        U = eigvec1_sorted.T @ eigvec2_sorted
        if not np.allclose(U @ U.T, np.eye(U.shape[0]), atol=atol):
            raise AssertionError("Eigenvectors should match up to an orthogonal transformation")

        tmp = eigvec1_sorted @ U
        diff = np.linalg.norm(tmp - eigvec2_sorted) / N
        assert diff < atol, "Eigenvectors should be identical up to an orthogonal transform"

def no_mixing(M: np.ndarray, v: np.ndarray):
    """
    Check that a 0-1 square matrix M does not connect states with different values in v.
    
    Parameters
    ----------
    M : np.ndarray
        Square matrix (0/1 entries).
    v : np.ndarray
        Vector of values/labels for each state.
        
    Raises
    ------
    AssertionError
        If any nonzero entry in M connects states with different labels.
    """
    assert M.shape[0] == M.shape[1], "Matrix must be square"
    assert M.shape[0] == v.shape[0], "Vector length must match matrix size"

    # Make a matrix of label comparisons
    same_label = v[:, None] == v[None, :]  # True if labels match

    # Check if any 1 in M connects different labels
    if np.any(M & (~same_label)):
        return False
    
    return True