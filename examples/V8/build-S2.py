
import os
import numpy as np
import pandas as pd
from functions import get_couplings
from QuantumSparse.operator import Operator
from QuantumSparse.spin import SpinOperators, compute_total_S2
from QuantumSparse.spin.functions import magnetic_moments, rotate_spins
from QuantumSparse.spin.interactions import Heisenberg, DM, anisotropy, rhombicity, Ising

# build spin operators
#print("\tbuilding spin operators ... ",end="")
S     = 1
NSpin = 8
spin_values = np.full(NSpin,S)
spins = SpinOperators(spin_values)

totS2 = compute_total_S2(spins)

print("\tn. blocks: {:d}".format(totS2.count_blocks(False))) 

# print("\tdiagonalizing the S2 operator ... ",end="") 
totS2.diagonalize()
# print("done")

folder = "output"
if not os.path.exists(folder): 
    os.mkdir(folder)
file = os.path.normpath("{:s}/S2.npz".format(folder))
print("\n\tsaving the S2 operator to file '{:s}' ... ".format(file),end="")  
totS2.save(file)
print("done")
      

print("\n\tJob done :)\n")

