import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, Heisenberg
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
def test_heisenberg_with_vs_without_symmetry(S, N):
    """
    Compare Heisenberg diagonalization with and without symmetries.

    Steps:
      - Check that the shift symmetry has the correct eigenvalues (roots of unity).
      - Diagonalize Heisenberg Hamiltonian with symmetry and without.
      - Ensure eigenvalues agree, and eigenstates match up to a unitary transformation.
    """
    spin_values = np.full(N, S)

    # construct the spin operators
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz

    #-----------------#
    # symmetry operator (translation / shift)
    D: Symmetry = shift(SpinOp)
    D.diagonalize(method="dense")
    l, n = D.energy_levels()

    assert len(l) == N, "wrong number of energy levels for shift symmetry"
    assert np.allclose(np.sort(l), np.sort(roots_of_unity(N))), "The eigenvalues should be the roots of unity."


    #-----------------#
    # Heisenberg Hamiltonian with some couplings
    H = Heisenberg(Sx, Sy, Sz, couplings=COUPLINGS)
    Hnosym = Operator(H.copy())  # independent copy

    assert np.all(H.data == Hnosym.data), "The Hamiltonians must match initially"
    assert H is not Hnosym, "Copies should be independent objects"

    # with symmetry
    H.diagonalize_with_symmetry(S=[D])
    
    # without symmetry
    Hnosym.diagonalize()
    
    # test
    compare_eigensolutions(H, Hnosym)
    
if __name__ == "__main__":
    pytest.main([__file__])