import pytest
import numpy as np
from quantumsparse.spin import Ising
from quantumsparse.statistics import correlation_function, Ising_sus_zz, T2beta
from quantumsparse.conftest import *

# ToDo: this works only for fews spins and for low temperatures
 
@parametrize_N
@pytest.mark.parametrize("S", [0.5])
def test_Ising_sus_zz(N,S):
    
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)

    H = Ising(Sz)
    H.diagonalize()
    check_diagonal(H)
    
    # Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)
    Sxtot = sum(Sx)
    
    T = np.logspace(np.log10(1), np.log10(100), 1000)
    beta = T2beta(T)
    chi = beta * correlation_function(T,H.eigenvalues,Sxtot,H.eigenstates)
    assert np.allclose(chi.imag, 0), "The susceptibility should be real."
    chi = chi.real
    assert np.all(chi >= 0), "The susceptibility should be non-negative."
    
    ref = Ising_sus_zz(N, S, T)
    
    assert np.allclose(chi/ref,1.0,atol=TOLERANCE), "The computed susceptibility does not match the reference solution."
    
    
if __name__ == "__main__":
    pytest.main([__file__])
