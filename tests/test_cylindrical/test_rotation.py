import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, Heisenberg
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.spin.functions import cylindrical_coordinates, rotate_spins, get_unitary_rotation_matrix, get_Euler_angles
from quantumsparse.tools.mathematics import product

TOLERANCE = 1e-10

@pytest.mark.parametrize("N, S", [
    (2, 0.5),
    (3, 0.5),
    (4, 0.5),
    (4, 1.0),
])
def test_rotation_method(N: int, S: float):
       
    spin_values = np.full(N, S)
    SpinOp = SpinOperators(spin_values)
    spins = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
    
    EulerAngles = get_Euler_angles(N)
    
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method="R")
    StU, SrU, SzU = rotate_spins(spins, EulerAngles=EulerAngles, method="U")
    
    for n in range(N):
        assert (StR[n] - StU[n]).norm() < TOLERANCE, "St rotation mismatch"
        assert (SrR[n] - SrU[n]).norm() < TOLERANCE, "Sr rotation mismatch"
        assert (SzR[n] - SzU[n]).norm() < TOLERANCE, "Sz rotation mismatch"

@pytest.mark.parametrize("method", ["R", "U"])
@pytest.mark.parametrize("N, S", [
    (2, 0.5),
    (3, 0.5),
    (4, 0.5),
    (4, 1.0),
])
def test_heisenberg_hamiltonian(N: int, S: float,method:str):
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
    spins = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
    Sx, Sy, Sz = spins
    
    U = cylindrical_coordinates(spins)

    # Spin operators in cartesian frame --> Heisenberg Hamiltonian in cartesian frame --> rotation to cylindrical frame
    H = Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=[1,2,3])
    H = H.unitary_transformation(U).clean()
    
    # Spin operators in cartesian frame --> rotation to cylindrical frame --> Hamiltonian in cylindrical frame
    EulerAngles = get_Euler_angles(N)
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method=method)
    Hcyl = Heisenberg(Sx=StR, Sy=SrR, Sz=SzR,couplings=[1,2,3])
    
    test = (Hcyl - H).norm()
    assert test < TOLERANCE, f"Heisenberg Hamiltonian mismatch in cylindrical coordinates (N={N}, S={S}): {test}"
    
    
    
    
    

