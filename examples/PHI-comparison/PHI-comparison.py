
import sys,os
dir_ = os.path.dirname(__file__)
os.chdir(dir_)
print("\n\tchanging working directory to:",dir_)
sys.path.append("../..")
import QuantumSparse as qs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


NLanczos = 200
tol = 1E-8
MaxDim=800
def diagonalize(H,NLanczos=NLanczos,tol=tol,MaxDim=MaxDim):
    return qs.diagonalize_Hamiltonian(H,NLanczos=NLanczos,tol=tol,MaxDim=MaxDim)


S     = 1./2.
NSpin = 8
SpinValues = np.full(NSpin,S)

Sx,Sy,Sz = qs.compute_spin_operators(SpinValues)
Mx,My,Mz = qs.magnetic_moment_operator(Sx,Sy,Sz)

# Hamiltonian [eV]
H  = qs.Heisenberg(Sx,Sy,Sz,couplings=1E-3,nn=1)
E,Psi = diagonalize(H)


spectrum = np.loadtxt("phi_levels.res") # cm^{-1}
spectrum *= 0.123983 # meV

fig,((ax00)) = plt.subplots(1,1,figsize=(10,5),sharey=False,sharex=False)
ax = ax00

ax.vlines((E-min(E))*1000,0.5,1,color="blue",label="qs")
ax.vlines(spectrum,0,0.5,color="red",label="PHI")

ax.set_yticks([],[])
ax.set_title("spectrum")
ax.set_xlabel("energy [meV]")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("spectrum.png")


B = 0.01
HB = H + qs.Zeeman(Sx,Sy,Sz,B=[0,0,B])
E,Psi = diagonalize(HB)

T = np.linspace(2,300,1000)
chi = qs.susceptibility(T,E,[Mz],[Mz],Psi)[0,0,:]

phi = pd.DataFrame(np.loadtxt("phi_sus.res"),columns=["T","chiT"])

fig,((ax00,ax01)) = plt.subplots(1,2,figsize=(12,5),sharey=False,sharex=False)
    
ax = ax00
ax.plot(T,chi*T,color="blue",label="qs")
ax.plot(phi["T"],phi["chiT"],color="red",label="PHI")
ax.set_ylabel("$\\chi$T [$cm^{3} mol^{-1} K$]")
ax.set_xlabel("T [K]")
ax.set_title("$\\chi$T")
ax.set_xlim(0,300)
ax.legend()
ax.grid(True)

ax = ax01
ax.plot(T,chi,color="blue",label="qs")
ax.plot(phi["T"],phi["chiT"]/phi["T"],color="red",label="PHI")
ax.set_ylabel("$\\chi$T [$cm^{3} mol^{-1}$]")
ax.set_xlabel("T [K]")
ax.set_title("$\\chi$")
ax.set_xlim(0,300)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("chi,chiT.png")

