import numpy as np
from copy import copy
import numba
from QuantumSparse import matrix
from typing import TypeVar, Union
# T = TypeVar('T') 

@numba.jit
def maxoff(x:matrix)->Union[float,int,int]:
    y = x.off_diagonal().tocoo()
    if len(y.data) == 0:
        return None, None, None
    y.data = abs(y.data)
    # indices = np.flip(np.argsort(y.data))[:Nrot] #np.argsort(y.data)[::-1][:Nrot]
    k = y.data.argmax()
    maxval = y.data[k] #[indices]
    maxrow = y.row[k] #[indices]
    maxcol = y.col[k] #[indices]
    return maxval, maxrow, maxcol

# @numba.jit
# def maxElemHermitian(matrix):
#     # Get the shape, data, indices, and indptr arrays of the CSR matrix
#     shape = matrix.shape
#     data = matrix.data
#     indices = matrix.indices
#     indptr = matrix.indptr

#     # Initialize variables to store the maximum off-diagonal value and its indices
#     max_value = None
#     max_row = None
#     max_col = None

#     # Iterate through the rows of the CSR matrix
#     for row in range(shape[0]):
#         start = indptr[row]
#         end = indptr[row + 1]

#         for i in range(start, end):
#             col = indices[i]

#             if col != row:  # Exclude diagonal elements
#                 if max_value is None or abs(data[i]) > abs(max_value):
#                     max_value = data[i]
#                     max_row = row
#                     max_col = col

#     return max_value, max_row, max_col

@numba.jit
def calculate_t(aii:float, ajj:float, aij:float)->float:
    numerator = 2 * abs(aij) * np.sign(aii - ajj)
    denominator = abs(aii - ajj) + np.sqrt( abs(aii - ajj)**2 + 4 * abs(aij)**2 )
    return numerator / denominator

@numba.jit
def calculate_cos_phi(t:float)->float:
    cos_phi = 1 / np.sqrt(1 + t**2)
    return cos_phi

@numba.jit
def calculate_sin_phi(t:float)->float:
    sin_phi = t / np.sqrt(1 + t**2)
    return sin_phi

@numba.jit
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

@numba.jit
def rotateHermitian(M:matrix, P:matrix, k:int, l:int)->Union[matrix,matrix]:

    G = Givens_rotation(M,k,l)

    # # if type(k) == np. and type(l) == list :
    # K = k
    # L = l
    # G = None
    # for k,l in zip(K,L):
    #     if G is None :
    #         G = Givens_rotation(M,k,l)
    #     else:
    #         G = G @ Givens_rotation(M,k,l) # apply on the 'external', on the 'right'

    # elif type(k) == int and type(l) == int:
    #     G = Givens_rotation(M,k,l)
    # else :
    #     raise ValueError("'k' and 'l' should have the same type")
    
    if not G.is_unitary():
        raise ValueError("Givens matrix is not unitary")
    
    Gdag = G.dagger()
    Mnew = Gdag @ M @ G
    Pnew = P @ G
    return Mnew, Pnew
 
# @numba.jit
def jacobi(M:matrix,tol:float=1.0e-3,max_iter:int=None)->Union[np.ndarray,matrix]:
    a = copy(M)
    n = len(a)
    max_iter = 5 * (n**2) if max_iter is None else max_iter  # Set limit on number of rotations
    # p = type(a)(a.shape,dtype=a.dtype)# np.eye(n, dtype=complex)  # Initialize transformation matrix
    p = a.identity((len(a)))

    for i in range(max_iter):
        aMax, k, l = maxoff(a)
        off_norm = a.off_diagonal().norm()
        line = "{:>6d}/{:<6d} | max: {:<12.6e} | off-norm: {:<12.6e} ".format(i,max_iter,aMax,off_norm)
        print(line,end="\r")
        # print("\t",a.count("off")," | aMax:", aMax,"  | off-norm:",off_norm,end="\r")
        if aMax is None or abs(off_norm) < tol:
            eigenvalues = a.diagonal()
            # if True:
            #     test = M @ p - p @ M.diags(eigenvalues,shape=M.shape)
            #     print("test:",test.norm())
            print('Jacobi method has converged\n')
            return eigenvalues, p

        a,p = rotateHermitian(a, p, k, l)

    print('Jacobi method reached the maximum number of iterations, i.e. {:d}'.format(max_iter))

    if M.count_blocks(False)  > 1 :
        return M.diagonalize()
    
    eigenvalues = a.diagonal()
    return eigenvalues, p