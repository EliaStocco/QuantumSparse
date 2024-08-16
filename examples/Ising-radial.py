from QuantumSparse.spin import SpinOperators
from QuantumSparse.operator import Operator
import numpy as np

S     = 2
NSpin = 4
spin_values = np.full(NSpin,S)

# construct the spin operators
SpinOp = SpinOperators(spin_values)
# unpack the operators
Sx,Sy,Sz = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz
# Sx is an array of length NSpin, 
print("\tSx.shape = ",Sx.shape)
# each element is a sparse matrix
# with the Sx operator acting on the corresponding site
print("\tSx[0].shape = ",Sx[0].shape)
# You will see that the shape of the operator mathces the size of the Hilbert space

# Good, these are your spin operators.
# Now you can build any operator that you want
# Let's build an Hamiltonian with radial 1nn interactions 
H =  Sy[0]@Sx[1] - Sx[1]@Sy[2] + Sy[2]@Sx[3] - Sx[3]@Sy[0] 

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