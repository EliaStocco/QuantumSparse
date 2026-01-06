import numpy as np
import pytest
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import cylindrical_coordinates, rotate_spins, get_Euler_angles
from quantumsparse.conftest import *

@parametrize_method
@parametrize_interaction
@parametrize_N
@parametrize_S
def test_hamiltonian(N: int, S: float,method:str,interaction:str):
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
    H = get_H(Sx=Sx, Sy=Sy, Sz=Sz, interaction=interaction)
    H = H.unitary_transformation(U)
    
    # Spin operators in cartesian frame --> rotation to cylindrical frame --> Hamiltonian in cylindrical frame
    EulerAngles = get_Euler_angles(N)
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method=method)
    Hcyl = get_H(Sx=StR, Sy=SrR, Sz=SzR, interaction=interaction)
    
    test = (Hcyl - H).norm()
    assert test < TOLERANCE, f"Heisenberg Hamiltonian mismatch in cylindrical coordinates (N={N}, S={S}): {test}"
    
if __name__ == "__main__":
    pytest.main([__file__])
    
    
    

