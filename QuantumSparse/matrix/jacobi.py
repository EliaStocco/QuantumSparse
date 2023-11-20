import numpy as np
from copy import copy
from QuantumSparse.tools.optimize import jit
from QuantumSparse import matrix
from typing import Union

@jit
def maxoff(x:matrix)->Union[float,int,int]:
    y = x.off_diagonal().tocoo()
    if len(y.data) == 0:
        return None, None, None
    y.data = abs(y.data)
    k = y.data.argmax()
    maxval = y.data[k]
    maxrow = y.row[k]
    maxcol = y.col[k]
    return maxval, maxrow, maxcol

@jit
def calculate_t(aii:float, ajj:float, aij:float)->float:
    numerator = 2 * abs(aij) * np.sign(aii - ajj)
    denominator = abs(aii - ajj) + np.sqrt( abs(aii - ajj)**2 + 4 * abs(aij)**2 )
    return numerator / denominator

@jit
def calculate_cos_phi(t:float)->float:
    cos_phi = 1 / np.sqrt(1 + t**2)
    return cos_phi

@jit
def calculate_sin_phi(t:float)->float:
    sin_phi = t / np.sqrt(1 + t**2)
    return sin_phi

@jit
def Givens_rotation(A:matrix, i:int, j:int)->matrix:
    # Get the elements at positions (i, j) and (j, i)
    aii = A[i, i]
    aij = A[i, j]
    aji = A[j, i]
    ajj = A[j, j]

    # t = calculate_t(aii,ajj,aij)
    # cosphi = calculate_cos_phi(t)
    # sinphi = calculate_sin_phi(t)
    phi = np.arctan2( 2 * abs(aij) , np.real(aii-ajj))/2
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    alpha = np.angle(aij)

    # Create the Givens rotation matrix (2x2)
    # Define the data (entries), row indices, and column indices
    data = np.array([                    cosphi, -np.exp(1.0j*alpha)*sinphi,
                     np.exp(-1.0j*alpha)*sinphi,                     cosphi])  # The non-zero values
    row_indices = np.array([i,i,j,j])  # Row indices for each value
    column_indices = np.array([i,j,i,j])  # Column indices for each value
    G = type(A)((data, (row_indices, column_indices)),shape=A.shape)
    rm = type(A)(([1,1], ([i,j], [i,j])),shape=A.shape)
    id = type(A).identity(len(A))
    out = G + id - rm
    return out

@jit
def rotateHermitian(M:matrix, P:matrix, k:int, l:int)->Union[matrix,matrix]:

    G = Givens_rotation(M,k,l)

    if not G.is_unitary():
        raise ValueError("Givens matrix is not unitary")
    
    Gdag = G.dagger()
    Mnew = Gdag @ M @ G
    Pnew = P @ G
    return Mnew, Pnew

@jit
def offRMSE(M):
    N = M.shape[0]*(M.shape[0]-1) # number of elements
    data = M.off_diagonal().data
    return np.sqrt(np.sum(np.square(np.absolute(data)))/N)
    
# @jit
def jacobi(M:matrix,tol:float=1.0e-3,max_iter:int=-1)->Union[np.ndarray,matrix,matrix]:
    a = copy(M) if M.nearly_diag is None else copy(M.nearly_diag)
    n = len(a)
    max_iter = 5 * (n**2) if max_iter < 0 else max_iter  # Set limit on number of rotations
    # p = type(a)(a.shape,dtype=a.dtype)# np.eye(n, dtype=complex)  # Initialize transformation matrix
    p = a.identity((len(a))) if M.eigenstates is None else copy(M.eigenstates)

    aMax, k, l = maxoff(a)
    #off_norm = a.off_diagonal().norm()
    rmse = offRMSE(a)
    # line = "Entering | max: {:<12.6e} | off RMSE: {:<12.6e} ".format(aMax,rmse)
    # print(line)

    for i in range(max_iter):
        aMax, k, l = maxoff(a)
        #off_norm = a.off_diagonal().norm()
        rmse = offRMSE(a)
        line = "{:>6d}/{:<6d} | max: {:<12.6e} | off RMSE: {:<12.6e} ".format(i,max_iter,aMax,rmse)
        print(line,end="\r")
        # print("\t",a.count("off")," | aMax:", aMax,"  | off-norm:",off_norm,end="\r")
        if aMax is None or rmse < tol:
            eigenvalues = a.diagonal()
            print('\nJacobi method has converged\n')
            return eigenvalues, p, a

        a,p = rotateHermitian(a, p, k, l)

    print('Jacobi method reached the maximum number of iterations, i.e. {:d}'.format(max_iter))
    
    eigenvalues = a.diagonal()
    return eigenvalues, p, a