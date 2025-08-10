import numpy as np
from quantumsparse.operator import Operator, Matrix

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
    Y:Matrix = Op @ Psi
    Y = Psi.conjugate().multiply(Y)
    Y = Y.real
    return np.array(Y.sum(axis=0)).flatten()


#@jit
def standard_deviation(Op:Operator,Psi:Operator,mean:np.ndarray=None)->np.ndarray:
    if mean is None :
        mean = expectation_value(Op,Psi)
    Quad = expectation_value(Op@Op,Psi)
    return np.sqrt( Quad - mean**2)



