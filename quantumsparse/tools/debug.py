import numpy as np
from quantumsparse.operator import Operator

def compare_eigensolutions(H1:Operator, H2:Operator, atol:float=1e-10,tol:float=1e-1)->None:
    
    test = H1.test_eigensolution().norm()
    assert test < tol, "Eigensolution of H1 not correct"
    
    test = H2.test_eigensolution().norm()
    assert test < tol, "Eigensolution of H2 not correct"
    
    assert H1.is_diagonalized(), "H1 is not diagonalized"
    assert H2.is_diagonalized(), "H2 is not diagonalized"
    assert H1.shape == H2.shape, "Operators must have the same shape"
    
    # sort eigenpairs
    _H1 = H1.sort()
    _H2 = H2.sort()
    assert np.allclose(_H1.eigenvalues, _H2.eigenvalues, atol=atol), \
        "Eigenvalues should be identical"

    # eigenstates can differ by a unitary transform (degenerate subspaces)
    diff = (_H1.eigenstates - _H2.eigenstates).norm()
    if diff > tol:
        U = _H1.eigenstates.dagger() @ _H2.eigenstates
        assert U.is_unitary(), "Eigenstates should match up to a unitary transformation"
        
