import numpy as np
import pytest
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin import SpinOperators, Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg, anisotropy, rhombicity
from quantumsparse.spin.functions import cylindrical_coordinates, rotate_spins, get_Euler_angles
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions
from quantumsparse.conftest import *

@parametrize_interaction
@parametrize_method
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
    
    def get_H(Sx,Sy,Sz)->Operator:
        if interaction == "heisenberg":
            H = Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=[1,2,3])
        elif interaction == "DM":
            H = Dzyaloshinskii_Moriya(Sx=Sx, Sy=Sy, Sz=Sz,couplings=[1,2,3])
        elif interaction == "biquadratic":
            H = biquadratic_Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=[1,2,3])
        elif interaction == "anisotropy":
            H = anisotropy(Sz=Sz)
        elif interaction == "rhombicity":
            H = rhombicity(Sx=Sx, Sy=Sy)
        else:
            raise ValueError(f"Unknown interaction: {interaction}")
        return H

    # Spin operators in cartesian frame --> Heisenberg Hamiltonian in cartesian frame --> rotation to cylindrical frame
    H = get_H(Sx=Sx, Sy=Sy, Sz=Sz)
    D = shift(SpinOp)
    D.diagonalize()
    H.diagonalize_with_symmetry([D])    
    H = H.unitary_transformation(U)
    
    # Spin operators in cartesian frame --> rotation to cylindrical frame --> Hamiltonian in cylindrical frame
    EulerAngles = get_Euler_angles(N)
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method=method)
    Hcyl = get_H(Sx=StR, Sy=SrR, Sz=SzR)
    Hcyl.diagonalize()
    
    test = (Hcyl - H).norm()
    assert test < TOLERANCE, f"Heisenberg Hamiltonian mismatch in cylindrical coordinates (N={N}, S={S}): {test}"
    
    compare_eigensolutions(H, Hcyl)
    
    Hcylsym = get_H(Sx=StR, Sy=SrR, Sz=SzR)
    Dcyl = D.unitary_transformation(U)
    Hcylsym.diagonalize_with_symmetry([Dcyl])
    compare_eigensolutions(Hcyl, Hcylsym)
    

if __name__ == "__main__":
    pytest.main([__file__])
    
    
    
    
    

