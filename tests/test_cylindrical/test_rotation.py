import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, Heisenberg
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.spin.functions import cylindrical_coordinates, rotate_spins, get_unitary_rotation_matrix, get_Euler_angles
from quantumsparse.tools.mathematics import product
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
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

    
if __name__ == "__main__":
    pytest.main([__file__])
    

