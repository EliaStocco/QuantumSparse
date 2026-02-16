import numpy as np
import pytest
from quantumsparse.operator import Operator
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
def test_spin_system_diagonalization(N, S):
    """Smoke test: construct spin operators, Zeeman Hamiltonian, and diagonalize."""
    spins = np.full(N,S)

    # Build spin operators
    SpinOp = SpinOperators(spins)
    _, _, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)

    # Dummy Hamiltonian (identity)
    H0 = Mz.empty()

    # Add simple Zeeman term
    B = 0.1
    H:Operator = H0 + Mz * B

    # Diagonalize (no symmetry for smoke test)
    H.diagonalize()
    check_diagonal(H)

    # Ensure eigenvalues exist and have correct length
    assert H.eigenvalues is not None
    assert len(H.eigenvalues) == int(2*S+1)**N

if __name__ == "__main__":
    pytest.main([__file__])