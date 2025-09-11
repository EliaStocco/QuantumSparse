import numpy as np
from quantumsparse.operator import Operator
from typing import Tuple

def compare_eigensolutions(H1:Operator, H2:Operator, atol:float=1e-10)->None:
    
    assert H1.shape == H2.shape, "Operators must have the same shape"
    assert H1.is_diagonalized(), "H1 is not diagonalized"
    assert H2.is_diagonalized(), "H2 is not diagonalized"
    
    N,M = H1.shape
    N = N*M
    
    test = H1.test_eigensolution().norm() / N 
    assert test < atol, "Eigensolution of H1 not correct"
    
    test = H2.test_eigensolution().norm() / N 
    assert test < atol, "Eigensolution of H2 not correct"
    
    # sort eigenpairs
    _H1 = H1.sort()
    _H2 = H2.sort()
    assert np.allclose(_H1.eigenvalues, _H2.eigenvalues, atol=atol), \
        "Eigenvalues should be identical"

    # eigenstates can differ by a unitary transform (degenerate subspaces)
    
    diff = (_H1.eigenstates - _H2.eigenstates).norm() / N
    if diff > atol:
        U:Operator = _H1.eigenstates.dagger() @ _H2.eigenstates
        assert U.is_unitary(), "Eigenstates should match up to a unitary transformation"
        
        tmp:Operator = _H1.eigenstates@U
        diff = (tmp - _H2.eigenstates).norm() / N
        assert diff < atol, "Eigenvectors should be identical"
        
# def compare_eigensolutions_dense_real(eigval1:np.ndarray,eigval2:np.ndarray,
#                                  eigvec1:np.ndarray,eigvec2:np.ndarray):
    
#     assert eigval1.shape == eigval2.shape
#     assert eigvec1.shape == eigvec2.shape
#     assert np.allclose(eigval1.imag,0)
#     assert np.allclose(eigval2.imag,0)
    
#     _eigval1 = np.sort(eigval1)
#     _eigval2 = np.sort(eigval2)
    
#     assert np.allclose(_eigval1,_eigval2)
    
#     diff = eigvec1 - eigvec2
#     norm = np.linalg.norm(diff)
#     if norm
    
    
        
