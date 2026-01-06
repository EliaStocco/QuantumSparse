import pytest
from quantumsparse.spin.functions import get_Euler_angles, rotate_spins
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
def test_rotation_method(N: int, S: float):
       
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)
    spins = (Sx, Sy, Sz)
    
    EulerAngles = get_Euler_angles(N)
    
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method="R")
    StU, SrU, SzU = rotate_spins(spins, EulerAngles=EulerAngles, method="U")
    
    for n in range(N):
        assert (StR[n] - StU[n]).norm() < TOLERANCE, "St rotation mismatch"
        assert (SrR[n] - SrU[n]).norm() < TOLERANCE, "Sr rotation mismatch"
        assert (SzR[n] - SzU[n]).norm() < TOLERANCE, "Sz rotation mismatch"

    
if __name__ == "__main__":
    pytest.main([__file__])
    

