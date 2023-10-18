
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
S     = 0.5
NSpin = 8
SpinValues = np.full(NSpin,S)
spins = spin_operators(SpinValues)

totS2 = spins.compute_total_S2()
S2 = spins.compute_S2()
#print("done")

# rotate spins
print("\n\tcomputing Euler's angles ... ",end="")
EulerAngles = np.zeros((8,3))
EulerAngles[:,2] = 360 - np.linspace(0,360,8,endpoint=False)
EulerAngles = np.pi * EulerAngles / 180  
#EulerAngles.fill(0)
print("done")

print("\trotating spins ... ",end="")           
# St,Sr,Sz= rotate_spins(spins=spins,EulerAngles=EulerAngles)
St,Sr,Sz = spins.Sx,spins.Sy,spins.Sz
print("done")

# load coupling constants
print("\treading coupling constants ... ",end="")           
couplings = get_couplings(S,"data","V8")
print("done")

# build the hamiltonian

dim = St[0].shape[0]
print("\tHilbert space dimension: {:d}".format(dim))   
print("\tbuilding the Hamiltonian: ")           
H = 0 # empty matrix
# print("\t\t         'empty': {:d} blocks".format(operator((St[0].shape)).count_blocks(False)))   
# H = Heisenberg(Sx=St,Sy=Sr,Sz=Sz)
# print("\t\t         'Ising': {:d} blocks".format(H.count_blocks(False)))  

H = operator((dim,dim))
H += Heisenberg(Sx=St,Sy=Sr,Sz=Sz,couplings=[couplings["Jt"],couplings["Jr"],couplings["Jz"]])
print("\t\t'Heisenberg 1nn': {:d} blocks".format(H.count_blocks(False)))     

# H += DM(Sx=St,Sy=Sr,Sz=Sz,couplings=[couplings["dt"],couplings["dr"],couplings["dz"]])
# print("\t\t            'DM': {:d} blocks".format(H.count_blocks(False)))      

# H += anisotropy(Sz=Sz,couplings=couplings["D"])
# print("\t\t    'anisotropy': {:d} blocks".format(H.count_blocks(False)))       

# H += rhombicity(Sx=St,Sy=Sr,couplings=couplings["E"])
# print("\t\t    'rhombicity': {:d} blocks".format(H.count_blocks(False)))       

# H += Heisenberg(Sx=St,Sy=Sr,Sz=Sz,nn=2,couplings=[couplings["Jt2"],couplings["Jr2"],couplings["Jz2"]])
# print("\t\t'Heisenberg 2nn': {:d} blocks".format(H.count_blocks(False)))       

# H  = H * 1E-3 # data are in meV, we build the Hamiltonian in eV
# print("done")

# H.count("all")
# H.count("off")
# H.count("diag")

w,f = H.diagonalize(tol=1)

# print("\tcounting Hamiltonian blocks: ",end="") 
# n,l = H.count_blocks(False)
# print("done")

folder = "output"
if not os.path.exists(folder): 
    os.mkdir(folder)
file = os.path.normpath("{:s}/Hamiltonian.npz".format(folder))
print("\tsaving the Hamiltonian to file '{:s}' ... ".format(file),end="")  
H.save(file)
print("done")

H = operator.load(file)

w,f = H.diagonalize(tol=1)

# M E = E L
# E-1 M E = L    

print("\n\tJob done :)\n")

