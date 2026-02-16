import pytest
import numpy as np
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.statistics import susceptibility, Curie_constant, T2beta
from quantumsparse.conftest import *

@parametrize_N
@parametrize_S
def test_Ising_sus_zz(N,S):
    
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)

    H = SpinOp.empty() # free spins
    H.diagonalize()
    check_diagonal(H)
    
    Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)
    
    T = np.logspace(np.log10(1), np.log10(300), 1000)
    beta  = T2beta(T)
    chiT = T*susceptibility(T,H,Mz)
    assert np.allclose(chiT.imag, 0), "The susceptibility should be real."
    chiT = chiT.real
    assert np.all(chiT >= 0), "The susceptibility should be non-negative."
    
    ref = Curie_constant(np.full(N,S))
    
    assert np.allclose(chiT, ref,atol=TOLERANCE), "The computed susceptibility does not match the reference solution."
    
    
if __name__ == "__main__":
    pytest.main([__file__])
