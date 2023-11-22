from QuantumSparse.operator import operator
import numpy as np

def mapping(N, i):
    if N % 2 == 0:
        # N is even
        return N - 1 - i
    else:
        # N is odd
        return i if i == 0 else N - i - 1

def flip(ops:operator)->operator:

    basis = np.asarray(ops.basis)
    D = ops.empty()
    N = basis.shape[1]
    left = np.full(N,np.nan)

    for c,right in enumerate(basis):
        for j in range(N):
            k = mapping(N,j)
            a = left[j]
            left[j] = left[k]
            left[k] = a
        # = np.concatenate((right[-1].reshape((1)), right[:-1]))
        r = np.where(np.all(basis == left, axis=1))
        D[r,c] = 1
    return D