import pytest
from quantumsparse.spin.functions import rotate_spins, get_Euler_angles
from quantumsparse.operator import Operator
from quantumsparse.tools.debug import check_commutation_relations
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
@parametrize_method
def test_commutation_relations(N: int, S: float, method:str) -> Operator:
    """
    Build a Hamiltonian for a ring of N spins of spin-S.
    """
    Sx, Sy, Sz, _ = NS2Ops(N, S)
    spins = (Sx, Sy, Sz)
    
    check_commutation_relations(Sx,Sy,Sz, TOLERANCE)
                
    EulerAngles = get_Euler_angles(N)
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method=method)
    check_commutation_relations(StR, SrR, SzR, TOLERANCE)

if __name__ == "__main__":
    pytest.main([__file__])