import numpy as np
from QuantumSparse.spin.spin_operators import spin_operators
from QuantumSparse.spin.interactions import Heisenberg
from QuantumSparse.spin.shift import shift

NSpin = 4
S = 0.5
spin_values = np.full(NSpin,S)
spins = spin_operators(spin_values)

D = shift(spins)

D.diagonalize(method="dense")



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

