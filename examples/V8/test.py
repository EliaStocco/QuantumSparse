
import os
import numpy as np
import pandas as pd
from functions import get_couplings
from QuantumSparse.operator import operator
from QuantumSparse.spin.spin_operators import spin_operators
from QuantumSparse.spin.functions import magnetic_moments, rotate_spins
from QuantumSparse.spin.interactions import Heisenberg, DM, anisotropy, rhombicity, Ising

# build spin operators
#print("\tbuilding spin operators ... ",end="")
S     = 1
NSpin = 8
spin_values = np.full(NSpin,S)
spins = spin_operators(spin_values)

totS2 = operator.load("output/S2.npz")#spins.compute_total_S2()
print("\tS2 blocks: {:d}".format(totS2.count_blocks(False)))       

H = Heisenberg(Sx=spins.Sx,Sy=spins.Sy,Sz=spins.Sz)
print("\tHeisenberg blocks: {:d}".format(H.count_blocks(False)))   

print(operator.commutator(totS2,H).norm()) # 1e-13

import numpy as np

# I should diagonalize the Hamiltonian using Sx,Sy,Sz
# Then I define the rotation matrix for the spins
# I then rotate the spins
# I then compute the susceptibility

def jacobi_rotation(A, tol=1e-6, max_iter=100):
    n = A.shape[0]
    eigenvalues = np.zeros(n)
    eigenvectors = np.eye(n)

    for kk in range(max_iter):
        max_off_diag = 0.0
        p, q = 0, 0

        # Find the maximum off-diagonal element
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_off_diag:
                    max_off_diag = abs(A[i, j])
                    p, q = i, j

        # Check for convergence
        if max_off_diag < tol:
            break

        # Compute the Jacobi rotation matrix
        theta = 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])
        c = np.cos(theta)
        s = np.sin(theta)

        # Apply the rotation to A and eigenvectors
        for i in range(n):
            A_ip = A[i, p]
            A_iq = A[i, q]
            A[p, i] = A[i, p] = c * A_ip - s * A_iq
            A[q, i] = A[i, q] = s * A_ip + c * A_iq

            eigenvectors_ip = eigenvectors[i, p]
            eigenvectors_iq = eigenvectors[i, q]
            eigenvectors[i, p] = c * eigenvectors_ip - s * eigenvectors_iq
            eigenvectors[i, q] = s * eigenvectors_ip + c * eigenvectors_iq

    for i in range(n):
        eigenvalues[i] = A[i, i]

    return eigenvalues, eigenvectors

# # Example usage
# A = np.array([[4.0, 3.0],
#               [3.0, 5.0]])

# eigenvalues, eigenvectors = jacobi_rotation(A)
# print("Eigenvalues:", eigenvalues)
# print("Eigenvectors:\n", eigenvectors)

w,f = jacobi_rotation(totS2,max_iter=1000)

print("\n\tJob done :)\n")

