import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.matrix import Matrix

def expectation_value(Op: Operator, Psi: Matrix) -> np.ndarray:
    """
    Compute <psi_i|Op|psi_i> for each column vector psi_i in Psi.

    Parameters
    ----------
    Op : csr_matrix
        Hermitian operator matrix (d × d).
    Psi : csr_matrix
        Matrix of state vectors as columns (d × N).

    Returns
    -------
    np.ndarray
        Expectation values array of length N.
    """
    Y = Op @ Psi
    Y = Psi.conjugate().multiply(Y)
    # Y = Matrix(Y)
    out = np.array(Y.sum(axis=0)).flatten()
    if Operator(Op).is_hermitean():
        assert np.allclose(out.imag, 0), "Expectation values should be real for a Hermitian operator."
        return out.real
    return out


#@jit
def standard_deviation(Op:Operator,Psi:Operator,mean:np.ndarray=None)->np.ndarray:
    if mean is None :
        mean = expectation_value(Op,Psi)
    Quad = expectation_value(Op@Op,Psi)
    return np.sqrt( Quad - mean**2)



