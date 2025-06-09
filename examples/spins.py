import numpy as np
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.spin import Ising

S     = 0.5
NSpin = 2
spin_values = np.full(NSpin,S)

# construct the spin operators
SpinOp = SpinOperators(spin_values)

# Let's have a look at the basis of the Hilbert space
print(SpinOp.basis)

# unpack the operators
Sx,Sy,Sz = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz

# Sx, Sy and Sz are np.arrays of Operators
# The Operator class inherit form Matrix

# Let's have a look at which elements are non-zero
Sx[0].visualize()

# or you can show some general information of the Operator
repr(Sx[0])

# or have a look at its elementa too
print(Sx[0])

# You can also convert it to a  usual numpy matrix
tmp = Sx[0].todense()
print(tmp)

# or numpy array
tmp = np.asarray(tmp)
print(tmp)



