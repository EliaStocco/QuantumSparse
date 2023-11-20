
import sys,os
dir_ = os.path.dirname(__file__)
os.chdir(dir_)
print("\n\tchanging working directory to:",dir_)
sys.path.append("../")
import QuantumSparse as qs
import numpy as np
import matplotlib.pyplot as plt


so = qs.spin_operators(S=1,N=4)


S     = 1
NSpin = 4
spin_values = np.full(NSpin,S)


Sx,Sy,Sz = qs.compute_spin_operators(spin_values)

# Hamiltonian 
H =  (Sx[0]@Sx[1]+Sy[0]@Sy[1]) + (Sy[1]@Sy[2]+Sx[1]@Sx[2]) +\
    (Sx[2]@Sx[3]+Sy[2]@Sy[3]) + (Sy[3]@Sy[0]+Sx[3]@Sx[0]) 
E0,Psi = qs.diagonalize_Hamiltonian(H,NLanczos=20,tol=1E-8,MaxDim=100)



Sx,Sy,Sz = qs.compute_spin_operators(spin_values)
EulerAngles = np.asarray([[0,0,0],\
               [0,0,-90],\
               [0,0,180],\
               [0,0,90]])    

EulerAngles = np.pi * EulerAngles / 180                     
St,Sr,Sz= qs.rotate(EulerAngles,Sx,Sy,Sz)

# Hamiltonian 
H = qs.DM(St,Sr,Sz,couplings=[0,0,1])
E1,Psi = qs.diagonalize_Hamiltonian(H,NLanczos=20,tol=1E-8,MaxDim=100)


fig,((ax0),(ax1)) = plt.subplots(2,1,figsize=(10,10),sharex=True)
axes = [ax0,ax1]

axes[0].vlines(E0,0,1,color="blue")
axes[0].grid(True)
axes[0].set_xlabel("energy [eV]")
axes[0].set_yticks([],[])
axes[0].set_title("Ising Hamiltonian: 1D chain with antisymmetric exchange interactions only")

axes[1].vlines(E1,0,1,color="red")
axes[1].grid(True)
axes[1].set_xlabel("energy [eV]")
axes[1].set_yticks([],[])
#axes[1].set_title("Heisenberg Hamiltonian")

plt.show()