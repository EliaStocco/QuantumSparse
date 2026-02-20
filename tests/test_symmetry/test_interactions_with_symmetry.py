import numpy as np
import pytest
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.statistics.statistical_physics import expectation_value
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
@parametrize_interaction
# @pytest.mark.parametrize("N", [4])
# @pytest.mark.parametrize("S", [1.0, 1.5, 2.0])
# @pytest.mark.parametrize("interaction",["Cr8-U.json"])
def test_dm_with_vs_without_symmetry(S, N, interaction):

    # spin operators
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)

    # symmetry operator (shift)
    D: Symmetry = shift(SpinOp)
    D.diagonalize()
    check_diagonal(D)
    l, n = D.energy_levels()
    assert len(l) == N, "wrong number of energy levels for shift symmetry"
    assert np.allclose(np.sort(l), np.sort(roots_of_unity(N))), "The eigenvalues should be the roots of unity."

    # Hamiltonian with Dzyaloshinskii–Moriya interaction
    H = get_H(Sx, Sy, Sz, interaction=interaction)
    H_not_diag = H.copy()

    # independent copy
    Hnosym = Operator(H.copy())

    # with symmetry
    H.diagonalize_with_symmetry(S=[D])
    # H = H.sort()
    check_diagonal(H)
    
    # without symmetry
    Hnosym.diagonalize()
    Hnosym = Hnosym.sort()
    check_diagonal(Hnosym)
    
    # test
    compare_eigensolutions(H, Hnosym)
    
    N,M = H.shape
    
    H_not_diag.eigenvalues = H.eigenvalues.copy()
    H_not_diag.eigenstates = H.eigenstates.copy()
    test = H_not_diag.test_eigensolution().norm() / N 
    assert test < TOLERANCE, "Eigensolution of hybrid(1) is not correct"
    
    H_not_diag.eigenvalues = Hnosym.eigenvalues.copy()
    H_not_diag.eigenstates = Hnosym.eigenstates.copy()
    test = H_not_diag.test_eigensolution().norm() / N 
    assert test < TOLERANCE, "Eigensolution of hybrid(2) is not correct"
    
    Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)
    
    T = np.asarray([0.,1., 10., 100.])
    exp_val1 = expectation_value(Mz, H.eigenstates)  # shape (N_states,)
    exp_val2 = expectation_value(Mz, Hnosym.eigenstates)  # shape (N_states,)
    MzH = H.thermal_average(operator=Mz, temperatures= T)
    MzHnosym = Hnosym.thermal_average(operator=Mz, temperatures= T)
    assert np.allclose(MzH, MzHnosym), "Thermal averages should be the same with and without symmetry."
    
    return
    

if __name__ == "__main__":
    pytest.main([__file__])