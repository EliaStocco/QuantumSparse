from typing import List
import numpy as np
from quantumsparse.operator import Operator

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
        
    hybrid = H1.copy()
    hybrid.eigenvalues = H2.eigenvalues.copy()
    hybrid.eigenstates = H2.eigenstates.copy()
    
    test = hybrid.test_eigensolution().norm() / N 
    assert test < atol, "Eigensolution of hybrid(1) is not correct"
    
    hybrid = H2.copy()
    hybrid.eigenvalues = H1.eigenvalues.copy()
    hybrid.eigenstates = H1.eigenstates.copy()
    
    test = hybrid.test_eigensolution().norm() / N 
    assert test < atol, "Eigensolution of hybrid(2) is not correct"
    

def compare_eigensolutions_dense_real(
    eigval1: np.ndarray,
    eigval2: np.ndarray,
    eigvec1: np.ndarray,
    eigvec2: np.ndarray,
    H: np.ndarray,
    atol: float = 1e-10,
) -> None:
    """
    Compare two eigensolutions (eigenvalues and eigenvectors) for real dense matrices.

    Eigenvectors are considered equal up to an orthogonal/unitary transformation
    within degenerate subspaces.

    Also verifies that the provided eigenvectors are indeed eigenvectors of H.

    Parameters
    ----------
    eigval1, eigval2 : np.ndarray
        Eigenvalue arrays (1D).
    eigvec1, eigvec2 : np.ndarray
        Eigenvector arrays (2D, columns are eigenvectors).
    H : np.ndarray
        Hamiltonian matrix (square, real/complex).
    atol : float
        Absolute tolerance for comparisons.
    """
    
    eigval1 = np.asarray(eigval1)
    eigval2 = np.asarray(eigval2)
    eigvec1 = np.asarray(eigvec1)
    eigvec2 = np.asarray(eigvec2)
    H = np.asarray(H)
    
    # ------------------------- #
    # Sanity checks
    assert eigval1.shape == eigval2.shape, "Eigenvalue arrays must have same shape"
    assert eigvec1.shape == eigvec2.shape, "Eigenvector arrays must have same shape"
    assert H.shape[0] == H.shape[1], "H must be square"
    assert eigvec1.shape[0] == H.shape[0], "Eigenvectors must match H dimensions"

    N = eigvec1.shape[0]

    # ------------------------- #
    # Check that each set of eigenvectors is indeed eigenvectors of H
    for vals, vecs, name in [(eigval1, eigvec1, "eigvec1"), (eigval2, eigvec2, "eigvec2")]:
        residual = np.linalg.norm(H @ vecs - vecs * vals[None, :]) / N
        assert residual < atol, f"{name} is not a valid eigenvector set (residual={residual})"

    # ------------------------- #
    # Sort eigenvalues and reorder eigenvectors accordingly
    idx1 = np.argsort(eigval1)
    idx2 = np.argsort(eigval2)
    eigval1_sorted = eigval1[idx1]
    eigval2_sorted = eigval2[idx2]
    eigvec1_sorted = eigvec1[:, idx1]
    eigvec2_sorted = eigvec2[:, idx2]

    # ------------------------- #
    # Compare eigenvalues
    assert np.allclose(eigval1_sorted, eigval2_sorted, atol=atol), \
        "Eigenvalues should be identical"
        
    # ------------------------- #
    # Check that each set of eigenvectors is indeed eigenvectors of H
    for vals, vecs, name in [(eigval1_sorted, eigvec1_sorted, "eigvec1_sorted"), (eigval2_sorted, eigvec2_sorted, "eigvec2_sorted")]:
        residual = np.linalg.norm(H @ vecs - vecs * vals[None, :]) / N
        assert residual < atol, f"{name} is not a valid eigenvector set (residual={residual})"

    # ------------------------- #
    # Compare eigenvectors
    diff = np.linalg.norm(eigvec1_sorted - eigvec2_sorted) / N

    if diff > atol:
        # Compute unitary/orthogonal alignment
        U = eigvec2_sorted @ eigvec1_sorted.conj().T
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=atol), \
            "U should be unitary"
        aligned = U @ eigvec1_sorted
        diff = np.linalg.norm(aligned - eigvec2_sorted) / N
        assert diff < atol, "Eigenvectors should match up to a unitary transformation"

        residual = np.linalg.norm(H @ aligned - aligned * eigval2_sorted[None, :]) / N
        assert residual < atol, f"Problem"
        
def check_commutation_relations(Sx:List[Operator],Sy:List[Operator],Sz:List[Operator],tolerance:float):
    N = len(Sx)
    for n in range(N):
        for m in range(N):
            if n != m:
                assert ( Sx[n].commutator(Sy[m]) ).norm() < tolerance, f"Commutation relation [Sx_n, Sy_m] != 0 failed at sites {n}, {m}"
                assert ( Sy[n].commutator(Sz[m]) ).norm() < tolerance, f"Commutation relation [Sy_n, Sz_m] != 0 failed at sites {n}, {m}"
                assert ( Sz[n].commutator(Sx[m]) ).norm() < tolerance, f"Commutation relation [Sz_n, Sx_m] != 0 failed at sites {n}, {m}"
            else:
                assert ( Sx[n].commutator(Sy[n]) - 1.j * Sz[n]).norm() < tolerance, f"Commutation relation [Sx, Sy] != i Sz failed at site {n}"
                assert ( Sy[n].commutator(Sz[n]) - 1.j * Sx[n]).norm() < tolerance, f"Commutation relation [Sy, Sz] != i Sx failed at site {n}"
                assert ( Sz[n].commutator(Sx[n]) - 1.j * Sy[n]).norm() < tolerance, f"Commutation relation [Sz, Sx] != i Sy failed at site {n}"