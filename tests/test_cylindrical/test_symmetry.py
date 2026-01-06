import pytest
from quantumsparse.spin.functions import cylindrical_coordinates, rotate_spins, get_Euler_angles
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
@parametrize_method
@parametrize_interaction
def test_symmetry(N: int, S: float,method:str,interaction:str):
    """
    Build a Hamiltonian for a ring of N spins of spin-S.

    Args:
        N (int): Number of spin sites.
        S (float): Spin value for each site.

    Returns:
        Operator: Hamiltonian as a sparse operator.
    """
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)
    spins = (Sx, Sy, Sz)
    
    U = cylindrical_coordinates(spins)

    # Spin operators in cartesian frame --> Hamiltonian in cartesian frame --> rotation to cylindrical frame
    H = get_H(Sx=Sx, Sy=Sy, Sz=Sz, interaction=interaction)
    D = shift(SpinOp)
    D.diagonalize()
    assert H.commute(D), "Hamiltonian does not commute with shift symmetry."
    assert D.commute(H), "Hamiltonian does not commute with shift symmetry."
    H.diagonalize_with_symmetry([D])    
    H = H.unitary_transformation(U)
    
    # Spin operators in cartesian frame --> rotation to cylindrical frame --> Hamiltonian in cylindrical frame
    EulerAngles = get_Euler_angles(N)
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method=method)
    Hcyl = get_H(Sx=StR, Sy=SrR, Sz=SzR, interaction=interaction)
    Hcyl.diagonalize()
    
    test = (Hcyl - H).norm()
    assert test < TOLERANCE, f"Hamiltonian mismatch in cylindrical coordinates (N={N}, S={S}): {test}"
    
    compare_eigensolutions(H, Hcyl)
    
    Hcylsym = get_H(Sx=StR, Sy=SrR, Sz=SzR,interaction=interaction)
    Dcyl = D.unitary_transformation(U)
    assert Hcylsym.commute(Dcyl), "Hamiltonian does not commute with shift symmetry."
    assert Dcyl.commute(Hcylsym), "Hamiltonian does not commute with shift symmetry."
    Hcylsym.diagonalize_with_symmetry([Dcyl])
    compare_eigensolutions(Hcyl, Hcylsym)
    compare_eigensolutions(H, Hcylsym)
    

if __name__ == "__main__":
    pytest.main([__file__])
    
    
    
    
    

