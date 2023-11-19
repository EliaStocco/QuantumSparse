import glob
import os
import numpy as np
import pandas as pd
from QuantumSparse.operator import operator
from QuantumSparse.spin.spin_operators import spin_operators
from QuantumSparse.spin.functions import magnetic_moments, rotate_spins
from QuantumSparse.spin.interactions import Heisenberg, DM, anisotropy, rhombicity, Ising

NSpin = 2
S = 0.5
SpinValues = np.full(NSpin,S)
spins = spin_operators(SpinValues)

totS2 = spins.compute_total_S2()
S2 = spins.compute_S2()

# build the hamiltonian
dim = spins.Sx[0].shape[0]
print("\tHilbert space dimension: {:d}".format(dim))   
print("\tbuilding the Hamiltonian: ")           

# H = operator((dim,dim))
H = Heisenberg(Sx=spins.Sx,Sy=spins.Sy,Sz=spins.Sz)
# H.visualize()
# H.count_blocks()
H.visualize()
print("\t\t'Heisenberg 1nn': {:d} blocks".format(H.count_blocks(False)))     

# diagonalize
# w,f = H.diagonalize(method="jacobi")

# test diagonalization
print("\n\tJob done :)\n")

