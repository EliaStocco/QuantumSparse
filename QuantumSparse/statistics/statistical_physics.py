# some function recalling statistical physics results
import numpy as np
from ..constants.constants import kB,g,_NA,_eV,muB
from QuantumSparse.tools.quantum_mechanics import expectation_value
from QuantumSparse import operator
import numba

@numba.jit
def T2beta(T):
    return 1.0/(kB*T)

@numba.jit
def partition_function(E,beta):
    return np.exp(-np.tensordot(beta,E-min(E),axes=0)).sum(axis=1)

@numba.jit
def classical_thermal_average_value(T:np.ndarray,E:np.ndarray,Obs:np.ndarray):
    beta = T2beta(T)
    Z = partition_function(E,beta)
    return (np.exp(-np.tensordot(beta,E-min(E),axes=0))*Obs).sum(axis=1)/Z

@numba.jit
def quantum_thermal_average_value(T,E,Op,Psi):
    Obs = expectation_value(Op,Psi)
    return classical_thermal_average_value(T,E,Obs)

@numba.jit
def correlation_function(T,E,OpAs,OpBs,Psi):
    # REWRITE THIS FUNCTION EXPLOITING quantum_mechanics.standard_deviation
    NT = len(T)    
    meanA =  np.zeros((len(OpAs),NT))       
    for n,Op in enumerate(OpAs):
        meanA[n] = quantum_thermal_average_value(T,E,Op,Psi)
    #    
    meanB =  np.zeros((len(OpBs),NT))
    for n,Op in enumerate(OpBs):
        meanB[n] = quantum_thermal_average_value(T,E,Op,Psi)             
    #
    square =  np.zeros((len(OpAs),len(OpBs),NT))
    for n1,OpA in enumerate(OpAs):
        for n2,OpB in enumerate(OpBs):        
            if n2 < n1 :
                square[n1,n2] =  square[n2,n1]
                continue
            
            square[n1,n2] = quantum_thermal_average_value(T,E,OpA@OpB,Psi)           
    #
    Chi =  np.zeros((3,3,NT))    
    for n1,OpA in enumerate(OpAs):
        for n2,OpB in enumerate(OpBs):            
            Chi[n1,n2] = (square[n1,n2] - meanA[n1]*meanB[n2])
    
    return Chi

@numba.jit
def susceptibility(T,E,OpAs,OpBs,Psi):
    beta  = 1.0/(kB*T)
    Chi = correlation_function(T,E,OpAs,OpBs,Psi) 
    return beta * Chi * _NA * _eV * 1E3  

@numba.jit
def Curie_constant(spin_values,gfactors=None):
    N = len(spin_values)
    if gfactors is None :
        gfactors = np.full(N,g)
    CW = np.zeros(N)
    for i in range(N):
        chi = gfactors[i]**2*muB**2*spin_values[i]*(spin_values[i]+1)/(3.*kB)
        CW[i] = _NA * _eV * 1E3  * chi 
    return CW.sum()
   