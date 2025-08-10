import numpy as np
import pytest
from scipy.sparse import random as sparse_random, eye as sparse_eye

# Assuming correlation_function is imported from your module
from quantumsparse.statistics import correlation_function
from quantumsparse.operator import Operator

@pytest.mark.parametrize("dim", [5, 10])
@pytest.mark.parametrize("num_temps", [3, 5])
def test_correlation_function_positive_when_OpB_none(dim, num_temps):
    # Generate temperature and energy arrays
    T = np.linspace(0.1, 10, num_temps)
    E = np.linspace(0, dim-1, dim)

    # Create random sparse Operator OpA (csr_matrix)
    OpA = sparse_random(dim, dim, density=0.3, format='csr', dtype=np.float64)
    # Make OpA symmetric (Hermitian) by averaging with its transpose
    OpA = (OpA + OpA.T) * 0.5
    OpA = Operator(OpA)

    # Create Psi as identity matrix (wavefunctions)
    Psi = sparse_eye(dim, format='csr')
    Psi = Operator(Psi)

    Chi = correlation_function(T, E, OpA, Psi, OpB=None)
    assert np.all(Chi >= 0), "Correlation function should be non-negative when OpB is None"
