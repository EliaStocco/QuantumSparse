import pytest
from quantumsparse.operator import Symmetry
from quantumsparse.spin.shift import shift
from quantumsparse.tools.debug import compare_eigensolutions
from quantumsparse.conftest import *
import time
import warnings


@parametrize_N
@parametrize_S
def test_math(S, N):

    # spin operators
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)
    
    # time this
    t0 = time.perf_counter()
    D1: Symmetry = shift(SpinOp,diagonalize=False)
    D1.diagonalize()
    t1 = time.perf_counter()
    
    # and this
    D2: Symmetry = shift(SpinOp,diagonalize=True)
    t2 = time.perf_counter()

    t_D1 = t1 - t0
    t_D2 = t2 - t1

    if t_D2 > t_D1:
        warnings.warn(
            f"D2 slower than D1: D1={t_D1:.4f}s, D2={t_D2:.4f}s",
            RuntimeWarning
        )
    
    compare_eigensolutions(D1,D2)
    
    
if __name__ == "__main__":
    pytest.main([__file__])