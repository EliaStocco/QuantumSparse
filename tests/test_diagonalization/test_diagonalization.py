import numpy as np
import pytest
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.tools.debug import compare_eigensolutions_dense_real
from quantumsparse.conftest import *

@parametrize_interaction
@parametrize_N
@parametrize_S
def test_hamiltonian_with_vs_without_blocks(N: int, S: float,interaction:str) -> Operator:
    """
    Build a Hamiltonian for a ring of N spins of spin-S.
    """
    spin_values = np.full(N, S)
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz

    H = get_H(Sx=Sx, Sy=Sy, Sz=Sz,interaction=interaction)
    eigenvalues, eigenvectors = H.diagonalize()
    
    Hdense = H.todense()
    eigenvalues_dense, eigenvectors_dense = np.linalg.eigh(Hdense)

    # test
    compare_eigensolutions_dense_real(eigenvalues,eigenvalues_dense,
                                 eigenvectors.todense(),eigenvectors_dense,
                                 Hdense,TOLERANCE)
    
if __name__ == "__main__":
    pytest.main([__file__])