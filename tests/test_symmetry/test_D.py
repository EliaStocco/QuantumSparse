import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, anisotropy
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions


@pytest.mark.parametrize("S,NSpin", [(0.5, 3), (1, 3), (1, 4)])
def test_anisotropy_with_vs_without_symmetry(S, NSpin):
    """
    Test that adding anisotropy to a Heisenberg Hamiltonian
    yields consistent results with and without symmetry diagonalization.
    """
    spin_values = np.full(NSpin, S)

    # spin operators
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz

    # symmetry operator (shift)
    D: Symmetry = shift(SpinOp)
    D.diagonalize(method="dense")
    l, N = D.energy_levels()
    assert np.allclose(np.sort(l), np.sort(roots_of_unity(NSpin)))

    # Hamiltonian with anisotropy
    H = anisotropy(Sz, couplings=1)  # easy-axis term

    # make independent copy
    Hnosym = Operator(H.copy())

    # with symmetry
    H.diagonalize_with_symmetry(S=[D])
    
    # without symmetry
    Hnosym.diagonalize()
    
    # test
    compare_eigensolutions(H, Hnosym)

if __name__ == "__main__":
    pytest.main([__file__])