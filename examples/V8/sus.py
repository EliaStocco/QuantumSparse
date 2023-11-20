
import os
import numpy as np
import pandas as pd
from functions import get_couplings
from QuantumSparse.operator import operator
from QuantumSparse.spin.spin_operators import spin_operators
from QuantumSparse.spin.functions import magnetic_moments, rotate_spins
   
#
# S     = 1.
# NSpin = 8
# spin_values = np.full(NSpin,S)
        
#
# Sx,Sy,Sz = spin_operators(spin_values)
S     = 1.
NSpin = 8
spin_values = np.full(NSpin,S)
spins = spin_operators(spin_values)

#

couplings = get_couplings(S,"data","V8")


# Mx,My,Mz = magnetic_moments(spins=spins)

print("\tcomputing Euler angles ... ",end="")
EulerAngles = np.zeros((8,3))
EulerAngles[:,2] = 360 - np.linspace(0,360,8,endpoint=False)
EulerAngles = np.pi * EulerAngles / 180
print("done")  

print("\tRotating spins ... ",end="")          
St,Sr,Sz= rotate_spins(spins=spins,EulerAngles=EulerAngles)
print("done") 

###
T = np.linspace(0.1,300,1000)
Barr = np.asarray([[0,0,0],[0,0,0.1],[0.1,0,0]])
#
cols = ["0T"]#,"perp_0.1T","par_0.1T"]
#
colsMag = [c + "-x" for c in cols] + [c + "-y" for c in cols] + [c + "-z" for c in cols]
Mag = np.zeros((3,len(T)))
magnetization = pd.DataFrame(np.zeros((len(T),len(colsMag)+1)),columns=["T"]+colsMag)
magnetization["T"] = T

###
colsX = list()
for i in ["x","y","z"]:
    for j in ["x","y","z"]:
        new = [c + "-" + i + j  for c in cols]
        for n in new:
            colsX.append(n)         
susceptibility = pd.DataFrame(np.zeros((len(T),len(colsX)+1)),columns=["T"]+colsX)
susceptibility["T"] = T

###
spectrum = pd.read_csv("data/spectrum.csv")
for B,col in zip(Barr,cols):
    print(B)
    Psi = np.load("data/eigenstates."+col+".npz")["arr_0"]
    E = spectrum[col]*1E-3
        
    # magnetization vs T
    Mag[0] = qs.quantum_thermal_average_value(T,E,Mx,Psi) / qs.muB
    Mag[1] = qs.quantum_thermal_average_value(T,E,My,Psi) / qs.muB
    Mag[2] = qs.quantum_thermal_average_value(T,E,Mz,Psi) / qs.muB
    
    magnetization[col+"-x"] = Mag[0]
    magnetization[col+"-y"] = Mag[1]
    magnetization[col+"-z"] = Mag[2]
    
    # diff. mag. sus vs T    
    Chi = qs.susceptibility(T,E,[Mx,My,Mz],[Mx,My,Mz],Psi)
    
    for ni,i in enumerate(["x","y","z"]):
        for nj,j in enumerate(["x","y","z"]):
            c = col+"-%s%s"%(i,j)
            susceptibility[c] = Chi[ni,nj]
    
###
magnetization.to_csv("data/magnetization.csv",index=False)
susceptibility.to_csv("data/susceptibility.csv",index=False)

###
col = "0T"
Psi = np.load("data/eigenstates."+col+".npz")["arr_0"]
E = spectrum[col]*1E-3

###
cols = ["t","r","z"]
OpsA = [St[0],Sr[0],Sz[0]]
OpsB = [St[0],Sr[0],Sz[0]]
#
corr = qs.correlation_function(T,E,OpsA,OpsB,Psi)     

corr_on_site = pd.DataFrame(columns=["T","tt","rr","zz","tr","rz","zt","rt","zr","tz"])
corr_on_site["T"] = T
for n,i in enumerate(cols):
    for m,j in enumerate(cols):
        ij = i+j
        corr_on_site.loc[:,ij] = corr[n,m]

corr_on_site.to_csv("data/corr_on_site.csv",index=False)

###
cols = ["t","r","z"]
OpsA = [St[0],Sr[0],Sz[0]]
OpsB = [St[1],Sr[1],Sz[1]]
#
corr = qs.correlation_function(T,E,OpsA,OpsB,Psi)     

corr_inter_site = pd.DataFrame(columns=["T","tt","rr","zz","tr","rz","zt","rt","zr","tz"])
corr_inter_site["T"] = T
for n,i in enumerate(cols):
    for m,j in enumerate(cols):
        ij = i+j
        corr_inter_site.loc[:,ij] = corr[n,m]

corr_inter_site.to_csv("data/corr_inter_site.csv",index=False)

###
cols = ["t","r","z"]
Ops = [St[0],Sr[0],Sz[0]]
#
mean = pd.DataFrame(columns=["T","t","r","z"])
mean["T"] = T
for i,Op in zip(cols,Ops):
    mean.loc[:,i] = qs.quantum_thermal_average_value(T,E,Op,Psi)

mean.to_csv("data/mean.csv",index=False)

###
cols = ["t","r","z"]
Ops = [St[0],Sr[0],Sz[0]]
#
square = pd.DataFrame(columns=["T","t","r","z"])
square["T"] = T
for i,Op in zip(cols,Ops):
    square.loc[:,i] = qs.quantum_thermal_average_value(T,E,Op@Op,Psi)

square.to_csv("data/square.csv",index=False)

###
print("\n\tFinished")