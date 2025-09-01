import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, Dzyaloshinskii_Moriya
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift


@pytest.mark.parametrize("S,NSpin", [(0.5, 3), (1, 3),(0.5,4),(1,4)])
def test_dm_with_vs_without_symmetry(S, NSpin):
    """
    Test that adding Dzyaloshinskii–Moriya interaction to a Heisenberg Hamiltonian
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

    # Hamiltonian with Dzyaloshinskii–Moriya interaction
    H = Dzyaloshinskii_Moriya(Sx, Sy, Sz, couplings=[1,2,3])

    # independent copy
    Hnosym = Operator(H.copy())

    # with symmetry
    E_sym, Psi_sym = H.diagonalize_with_symmetry(S=[D])
    assert H.test_eigensolution().norm() < 1e-10

    # without symmetry
    E_plain, Psi_plain = Hnosym.diagonalize()
    assert Hnosym.test_eigensolution().norm() < 1e-10

    # compare eigenvalues
    assert np.allclose(np.sort(E_sym.real), np.sort(E_plain.real), atol=1e-10)

    # eigenstates: should agree up to a unitary
    diff = (H.eigenstates - Hnosym.eigenstates).norm()
    if diff > 1e-10:
        U = H.eigenstates.dagger() @ Hnosym.eigenstates
        assert U.is_unitary()
