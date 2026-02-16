import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, anisotropy
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
def test_anisotropy_with_vs_without_symmetry(S, N):
    """
    Test that adding anisotropy to a Heisenberg Hamiltonian
    yields consistent results with and without symmetry diagonalization.
    """

    # spin operators
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)

    # symmetry operator (shift)
    D: Symmetry = shift(SpinOp)
    D.diagonalize()
    check_diagonal(D)
    l, n = D.energy_levels()
    assert len(l) == N, "wrong number of energy levels for shift symmetry"
    assert np.allclose(np.sort(l), np.sort(roots_of_unity(N))), "The eigenvalues should be the roots of unity."

    # Hamiltonian with anisotropy
    H = anisotropy(Sz, couplings=COUPLINGS[0])  # easy-axis term

    # make independent copy
    Hnosym = Operator(H.copy())

    # with symmetry
    H.diagonalize_with_symmetry(S=[D])
    check_diagonal(H)
    
    # without symmetry
    Hnosym.diagonalize()
    check_diagonal(Hnosym)
    
    # test
    compare_eigensolutions(H, Hnosym)

if __name__ == "__main__":
    pytest.main([__file__])