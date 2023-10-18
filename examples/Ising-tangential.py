
import sys,os
dir_ = os.path.dirname(__file__)
os.chdir(dir_)
print("\n\tchanging working directory to:",dir_)
sys.path.append("..")
import QuantumSparse as qs
from QuantumSparse.spin_operators import spin_operators as so
import numpy as np
import matplotlib.pyplot as plt


S     = 5
NSpin = 4
SpinValues = np.full(NSpin,S)


Sx,Sy,Sz = so.compute_spin_operators(SpinValues)

# Hamiltonian 
H = - Sx[0]@Sy[1] + Sy[1]@Sx[2] - Sx[2]@Sy[3] + Sy[3]@Sx[0] 
E0,Psi = qs.diagonalize_Hamiltonian(H,NLanczos=20,tol=1E-8,MaxDim=100)
E0 = E0.real
E0.sort()
E0 = E0-min(E0)
#print(E)



Sx,Sy,Sz = qs.compute_spin_operators(SpinValues)
EulerAngles = np.asarray([[0,0,0],\
               [0,0,-90],\
               [0,0,180],\
               [0,0,90]])    

EulerAngles = np.pi * EulerAngles / 180                     
St,Sr,Sz= qs.rotate(EulerAngles,Sx,Sy,Sz)

# Hamiltonian 
#H = St[0]@St[1] + St[1]@St[2] + St[2]@St[3] + St[3]@St[0] 
H = qs.Ising(St)
E1,Psi = qs.diagonalize_Hamiltonian(H,NLanczos=20,tol=1E-8,MaxDim=100)


fig,((ax0),(ax1)) = plt.subplots(2,1,figsize=(10,10),sharex=True)
axes = [ax0,ax1]

axes[0].vlines(E0,0,1,color="blue")
axes[0].grid(True)
axes[0].set_xlabel("energy [eV]")
axes[0].set_yticks([],[])
axes[0].set_title("Ising Hamiltonian: 1D chain with tangential spin interactions only")

axes[1].vlines(E1,0,1,color="red")
axes[1].grid(True)
axes[1].set_xlabel("energy [eV]")
axes[1].set_yticks([],[])
#axes[1].set_title("Heisenberg Hamiltonian")

plt.show()