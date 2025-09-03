import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from quantumsparse.tools.quantum_mechanics import expectation_value

@pytest.mark.parametrize("dim,N", [
    (4, 2),
    (6, 5),
    (8, 3),
    (10, 7),
])
def test_expectation_value_random_operator_and_vectors(dim, N):
    """
    Test expectation values with random Hermitian operator and random vectors.
    Checks that results are real and consistent.
    """
    np.random.seed(0)
    
    # Random Hermitian operator
    A = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    Op_dense = (A + A.conj().T) / 2
    Op = csr_matrix(Op_dense)
    
    # Random normalized vectors as columns of Psi
    Psi_dense = np.random.randn(dim, N) + 1j*np.random.randn(dim, N)
    Psi_dense /= np.linalg.norm(Psi_dense, axis=0)
    Psi = csr_matrix(Psi_dense)
    
    ev = expectation_value(Op, Psi)
    assert ev.dtype == np.float64 or ev.dtype == float, "Expectation values should be real."
    assert np.all(np.abs(ev.imag) < 1e-12), "Imaginary parts should be negligible."
    
    eigvals = np.linalg.eigvalsh(Op_dense)
    assert np.all(ev >= eigvals.min() - 1e-12) and np.all(ev <= eigvals.max() + 1e-12), \
        "Expectation values should be within eigenvalue range."

if __name__ == "__main__":
    pytest.main([__file__])