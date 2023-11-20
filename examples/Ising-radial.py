
import sys,os
dir_ = os.path.dirname(__file__)
os.chdir(dir_)
print("\n\tchanging working directory to:",dir_)
sys.path.append("..")
import QuantumSparse as qs
import numpy as np
import matplotlib.pyplot as plt


S     = 2
NSpin = 4
spin_values = np.full(NSpin,S)


Sx,Sy,Sz = qs.compute_spin_operators(spin_values)

# Hamiltonian 
H =  Sy[0]@Sx[1] - Sx[1]@Sy[2] + Sy[2]@Sx[3] - Sx[3]@Sy[0] 
E0,Psi = qs.diagonalize_Hamiltonian(H,NLanczos=20,tol=1E-8,MaxDim=100)
E0 = E0.real
E0.sort()
E0 = E0-min(E0)
#print(E)



Sx,Sy,Sz = qs.compute_spin_operators(spin_values)
EulerAngles = np.asarray([[0,0,0],\
               [0,0,-90],\
               [0,0,180],\
               [0,0,90]])    

EulerAngles = np.pi * EulerAngles / 180                     
St,Sr,Sz= qs.rotate(EulerAngles,Sx,Sy,Sz)

# Hamiltonian 
#H = Sr[0]@Sr[1] + Sr[1]@Sr[2] + Sr[2]@Sr[3] + Sr[3]@Sr[0] 
H = qs.Ising(Sr)
E1,Psi = qs.diagonalize_Hamiltonian(H,NLanczos=20,tol=1E-8,MaxDim=100)


fig,((ax0),(ax1)) = plt.subplots(2,1,figsize=(10,10),sharex=True)
axes = [ax0,ax1]

axes[0].vlines(E0,0,1,color="blue")
axes[0].grid(True)
axes[0].set_xlabel("energy [eV]")
axes[0].set_yticks([],[])
axes[0].set_title("Ising Hamiltonian: 1D chain with radial spin interactions only")

axes[1].vlines(E1,0,1,color="red")
axes[1].grid(True)
axes[1].set_xlabel("energy [eV]")
axes[1].set_yticks([],[])
#axes[1].set_title("Heisenberg Hamiltonian")

plt.show()