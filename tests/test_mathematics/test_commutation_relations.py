import pytest
from quantumsparse.spin.functions import rotate_spins, get_Euler_angles
from quantumsparse.operator import Operator
from quantumsparse.conftest import *

def commutation_relations(Sx:List[Operator],Sy:List[Operator],Sz:List[Operator]):
    N = len(Sx)
    for n in range(N):
        for m in range(N):
            if n != m:
                assert ( Sx[n].commutator(Sy[m]) ).norm() < TOLERANCE, f"Commutation relation [Sx_n, Sy_m] != 0 failed at sites {n}, {m}"
                assert ( Sy[n].commutator(Sz[m]) ).norm() < TOLERANCE, f"Commutation relation [Sy_n, Sz_m] != 0 failed at sites {n}, {m}"
                assert ( Sz[n].commutator(Sx[m]) ).norm() < TOLERANCE, f"Commutation relation [Sz_n, Sx_m] != 0 failed at sites {n}, {m}"
            else:
                assert ( Sx[n].commutator(Sy[n]) - 1.j * Sz[n]).norm() < TOLERANCE, f"Commutation relation [Sx, Sy] != i Sz failed at site {n}"
                assert ( Sy[n].commutator(Sz[n]) - 1.j * Sx[n]).norm() < TOLERANCE, f"Commutation relation [Sy, Sz] != i Sx failed at site {n}"
                assert ( Sz[n].commutator(Sx[n]) - 1.j * Sy[n]).norm() < TOLERANCE, f"Commutation relation [Sz, Sx] != i Sy failed at site {n}"

@parametrize_N
@parametrize_S
@parametrize_method
def test_commutation_relations(N: int, S: float, method:str) -> Operator:
    """
    Build a Hamiltonian for a ring of N spins of spin-S.
    """
    Sx, Sy, Sz, _ = NS2Ops(N, S)
    spins = (Sx, Sy, Sz)
    
    commutation_relations(Sx,Sy,Sz)
                
    EulerAngles = get_Euler_angles(N)
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method=method)
    commutation_relations(StR, SrR, SzR)

if __name__ == "__main__":
    pytest.main([__file__])