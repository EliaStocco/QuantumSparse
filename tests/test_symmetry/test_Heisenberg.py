import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, Heisenberg
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions

@pytest.mark.parametrize("N, S", [
    (2, 0.5),
    (3, 0.5),
    (4, 0.5),
    (4, 1.0),
])
def test_heisenberg_hamiltonian(N: int, S: float) -> Operator:
    """
    Build a Heisenberg Hamiltonian for a ring of N spins of spin-S.

    Args:
        N (int): Number of spin sites.
        S (float): Spin value for each site.

    Returns:
        Operator: Heisenberg Hamiltonian as a sparse operator.
    """
    spin_values = np.full(N, S)
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz

    # Construct Heisenberg Hamiltonian manually
    H_manual:Operator = sum(Sx[i] @ Sx[(i + 1) % N] for i in range(N))
    H_manual += sum(Sy[i] @ Sy[(i + 1) % N] for i in range(N))
    H_manual += sum(Sz[i] @ Sz[(i + 1) % N] for i in range(N))
    # H_manual = Operator(H_manual)

    # Compare with library's Heisenberg Hamiltonian
    H_lib = Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz)
    assert np.allclose(H_lib.todense(), H_manual.todense()), "Mismatch in Heisenberg Hamiltonian construction"
    
@pytest.mark.parametrize("S,NSpin", [(0.5, 3), (1, 4), (1.5, 2)])
def test_heisenberg_with_vs_without_symmetry(S, NSpin):
    """
    Compare Heisenberg diagonalization with and without symmetries.

    Steps:
      - Check that the shift symmetry has the correct eigenvalues (roots of unity).
      - Diagonalize Heisenberg Hamiltonian with symmetry and without.
      - Ensure eigenvalues agree, and eigenstates match up to a unitary transformation.
    """
    spin_values = np.full(NSpin, S)

    # construct the spin operators
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz

    #-----------------#
    # symmetry operator (translation / shift)
    D: Symmetry = shift(SpinOp)
    D.diagonalize(method="dense")
    l, N = D.energy_levels()

    assert len(l) == NSpin, "wrong number of energy levels for shift symmetry"
    ru = np.sort(roots_of_unity(len(spin_values)))
    assert np.allclose(np.sort(l), ru), "The eigenvalues should be the roots of unity."

    #-----------------#
    # Heisenberg Hamiltonian with some couplings
    H = Heisenberg(Sx, Sy, Sz, couplings=[1, 2, 3])
    Hnosym = Operator(H.copy())  # independent copy

    assert np.all(H.data == Hnosym.data), "The Hamiltonians must match initially"
    assert H is not Hnosym, "Copies should be independent objects"

    # with symmetry
    H.diagonalize_with_symmetry(S=[D])
    
    # without symmetry
    Hnosym.diagonalize()
    
    # test
    compare_eigensolutions(H, Hnosym)