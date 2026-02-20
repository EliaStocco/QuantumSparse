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
        
        # R and U are the same
        assert (StR[n] - StU[n]).norm() < TOLERANCE, "St rotation mismatch"
        assert (SrR[n] - SrU[n]).norm() < TOLERANCE, "Sr rotation mismatch"
        assert (SzR[n] - SzU[n]).norm() < TOLERANCE, "Sz rotation mismatch"
        
        # St and Sr change
        assert not (n>0 and (StR[n] - Sx[n]).norm() < TOLERANCE), "StR rotation mismatch"
        assert not (n>0 and (StU[n] - Sx[n]).norm() < TOLERANCE), "StU rotation mismatch"
        assert not (n>0 and (SrR[n] - Sy[n]).norm() < TOLERANCE), "SrR rotation mismatch"
        assert not (n>0 and (SrU[n] - Sy[n]).norm() < TOLERANCE), "SrU rotation mismatch"
        
        # Sz does not change
        assert (SzR[n] - Sz[n]).norm() < TOLERANCE, "Sz rotation mismatch"
        assert (SzU[n] - Sz[n]).norm() < TOLERANCE, "Sz rotation mismatch"
        
    return

    
if __name__ == "__main__":
    pytest.main([__file__])
    

