import numpy as np
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.spin import Ising

# In quantumsparse/spin/interactions.py you can find:
# - Ising
# - Heisenberg
# - DM
# - anisotropy
# - rhombicity
# - BiQuadraticIsing
# - BiQuadraticHeisenberg

S     = 2
NSpin = 4
spin_values = np.full(NSpin,S)

# construct the spin operators
SpinOp = SpinOperators(spin_values)
# unpack the operators
Sx,Sy,Sz = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz

# Ising Hamiltonian along the z-axis
H = Ising(Sz) 
print("\tH.shape = ",H.shape)

# Let's build the Ising Hamiltonian from scratch
Htest = Sz[0]@Sz[1] + Sz[1]@Sz[2] + Sz[2]@Sz[3] + Sz[3]@Sz[0] 

# This line should not be necessary
Htest = Operator(Htest)

assert np.allclose(H.todense(),Htest.todense()), "The Hamiltonians should be the same"

# Just for static typing reason ... 
print("\ttype(H):",type(H))
H = Operator(H)
# Let's have a look at the Hamiltonian
repr(H) # better than print(H)

# Diagonalize!
# YOU CAN DECOMMENT THESE LINES
# print()
# E0,Psi = H.diagonalize(method="dense") #),NLanczos=20,tol=1E-8,MaxDim=100)

# or let's do with our favorite numerical routine
Hdense = np.asarray(H.todense()) # let's convert H from a scipy.sparse matrix into a normal np.array
print("\ttype(Hdense):",type(Hdense))
E0,Psi = np.linalg.eigh(Hdense)

E0 = E0.real
E0.sort()

print("\tmin eigenvalue:",E0[0])
print("\tmax eigenvalue:",E0[-1])
E0 = E0-min(E0)
print("\tenergy range:",E0[-1]-E0[0])

pass