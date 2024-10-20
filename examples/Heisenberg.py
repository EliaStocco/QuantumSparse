import numpy as np
from QuantumSparse.spin import SpinOperators
from QuantumSparse.operator import Operator
from QuantumSparse.spin import Heisenberg, DM

from scipy.sparse.linalg import eigsh, eigs

# In QuantumSparse/spin/interactions.py you can find:
# - Ising
# - Heisenberg
# - DM
# - anisotropy
# - rhombicity
# - BiQuadraticIsing
# - BiQuadraticHeisenberg

S     = 0.5 # spin value for all sites
NSpin = 4 # number of spins
spin_values = np.full(NSpin,S) # spin values for all the sites

# spin_values = np.asarray([0.5,0.5,1,0.5])

# construct the spin operators
SpinOp = SpinOperators(spin_values)
# unpack the operators
Sx,Sy,Sz = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz

# Ising Hamiltonian along the z-axis
H = Heisenberg(Sx=Sx,Sy=Sy,Sz=Sz)
# dm_term = DM(Sx,Sy,Sz)
# H = H + dm_term 
print("\tH.shape = ",H.shape)

# Let's build the Heisenberg Hamiltonian from scratch
x = Sx[0]@Sx[1] + Sx[1]@Sx[2] + Sx[2]@Sx[3] + Sx[3]@Sx[0] # Ising interaction along x
y = Sy[0]@Sy[1] + Sy[1]@Sy[2] + Sy[2]@Sy[3] + Sy[3]@Sy[0] # Ising interaction along y
z = Sz[0]@Sz[1] + Sz[1]@Sz[2] + Sz[2]@Sz[3] + Sz[3]@Sz[0] # Ising interaction along z
Htest = Operator(x+y+z) # And this is the Heisenberg Hamiltonian

assert np.allclose(H.todense(),Htest.todense()), "The Hamiltonians should be the same"

# Just for static typing reason ... 
print("\ttype(H):",type(H))
H = Operator(H)
# Let's have a look at the Hamiltonian
repr(H) # better than print(H)

w,f = eigs(H,k=5)

# Diagonalize!
# YOU CAN DECOMMENT THESE LINES
# print()
E0,Psi = H.diagonalize() #),NLanczos=20,tol=1E-8,MaxDim=100)

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