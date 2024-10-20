# some function recalling statistical physics results
import numpy as np
from QuantumSparse.constants import kB,g,_NA,_eV,muB
from QuantumSparse.tools.quantum_mechanics import expectation_value
from QuantumSparse.operator import Operator
from QuantumSparse.matrix import Matrix


def T2beta(T:np.ndarray)->np.ndarray:
    """
    Calculates the inverse temperature array from a given temperature array.

    Parameters:
        T (np.ndarray): The temperature array.

    Returns:
        np.ndarray: The inverse temperature array.
    """
    return 1.0/(kB*T)


def partition_function(E: np.ndarray,beta:np.ndarray)->np.ndarray:
    """
    Calculates the partition function of a system.

    Parameters:
        E (np.ndarray): The energy array.
        beta (np.ndarray): The inverse temperature array.

    Returns:
        np.ndarray: The partition function of the system.
    """
    return np.exp(-np.tensordot(beta,E-min(E),axes=0)).sum(axis=1)


def classical_thermal_average_value(T:np.ndarray,E:np.ndarray,Obs:np.ndarray)->np.ndarray:
    """
    Calculate the classical thermal average value of an observable in a system.

    Parameters:
        T (np.ndarray): The temperature array.
        E (np.ndarray): The energy array.
        Obs (np.ndarray): The observable array.

    Returns:
        np.ndarray: The classical thermal average value of the observable.
    """
    beta = T2beta(T)
    Z = partition_function(E,beta)
    return (np.exp(-np.tensordot(beta,E-min(E),axes=0))*Obs).sum(axis=1)/Z


def quantum_thermal_average_value(T: np.ndarray,E: np.ndarray,Op:Operator,Psi:Matrix)->np.ndarray:
    """
    Calculates the quantum thermal average value of an observable in a system.

    Parameters
    ----------
    T : np.ndarray
        The temperature array.
    E : np.ndarray
        The energy array.
    Op : Operator
        The operator representing the observable.
    Psi : Matrix
        The wave function of the system.

    Returns
    -------
    np.ndarray
        The quantum thermal average value of the observable.
    """
    Obs = expectation_value(Op,Psi)
    return classical_thermal_average_value(T,E,Obs)

def correlation_function(T: np.ndarray, E: np.ndarray, OpA: Operator, Psi: Matrix, OpB: Operator=None) -> np.ndarray:
    """
    Calculates the correlation function of a system.

    Parameters
    ----------
    T : np.ndarray
        The temperature array.
    E : np.ndarray
        The energy array.
    OpA : Operator
        The first operator.
    OpB : Operator
        The second operator.
    Psi : Matrix
        The wave function of the system.

    Returns
    -------
    np.ndarray
        The correlation function of the system.
    """
    # Calculate the number of temperature points
    NT = len(T)
    
    # Compute the thermal averages for the operators
    meanA = quantum_thermal_average_value(T, E, OpA, Psi)
    if OpB is not None:
        meanB = quantum_thermal_average_value(T, E, OpB, Psi)
        OpAB = OpA @ OpB
    else:
        meanB = meanA
        OpAB = OpA @ OpA
    
    # Calculate the correlation function
    square = quantum_thermal_average_value(T, E, OpAB, Psi)
    
    # Initialize the Chi array to hold the correlation result
    Chi = square - meanA * meanB
    return Chi


# def correlation_function(T: np.ndarray,E: np.ndarray,OpAs: Operator,OpBs: Operator,Psi:Matrix)->np.ndarray:
#     """
#     Calculates the correlation function of a system.

#     Parameters
#     ----------
#     T : np.ndarray
#         The temperature array.
#     E : np.ndarray
#         The energy array.
#     OpAs : Operator
#         The first set of operators.
#     OpBs : Operator
#         The second set of operators.
#     Psi : Matrix
#         The wave function of the system.

#     Returns
#     -------
#     np.ndarray
#         The correlation function of the system.
#     """
#     # REWRITE THIS FUNCTION EXPLOITING quantum_mechanics.standard_deviation
#     NT = len(T)    
#     meanA =  np.zeros((len(OpAs),NT))       
#     for n,Op in enumerate(OpAs):
#         meanA[n] = quantum_thermal_average_value(T,E,Op,Psi)
#     #    
#     meanB =  np.zeros((len(OpBs),NT))
#     for n,Op in enumerate(OpBs):
#         meanB[n] = quantum_thermal_average_value(T,E,Op,Psi)             
#     #
#     square =  np.zeros((len(OpAs),len(OpBs),NT))
#     for n1,OpA in enumerate(OpAs):
#         for n2,OpB in enumerate(OpBs):        
#             if n2 < n1 :
#                 square[n1,n2] =  square[n2,n1]
#                 continue
            
#             square[n1,n2] = quantum_thermal_average_value(T,E,OpA@OpB,Psi)           
#     #
#     Chi =  np.zeros((3,3,NT))    
#     for n1,OpA in enumerate(OpAs):
#         for n2,OpB in enumerate(OpBs):            
#             Chi[n1,n2] = (square[n1,n2] - meanA[n1]*meanB[n2])
    
#     return Chi


def susceptibility(T: np.ndarray,H:Operator,OpAs:Operator,OpBs:Operator=None)->np.ndarray:
    """
    Calculates the magnetic susceptibility of a system.

    Parameters
    ----------
    T : np.ndarray
        The temperature array.
    H : Operator
        The Hamiltonian operator of the system.
    OpAs : Operator
        The first set of operators.
    OpBs : Operator
        The second set of operators.

    Returns
    -------
    np.ndarray
        The magnetic susceptibility of the system.
    """
    beta  = 1.0/(kB*T)
    Chi = correlation_function(T,H.eigenvalues-np.min(H.eigenvalues),OpAs,H.eigenstates,OpBs) 
    return beta * Chi * _NA * _eV * 1E3  


def Curie_constant(spin_values,gfactors=None):
    N = len(spin_values)
    if gfactors is None :
        gfactors = np.full(N,g)
    CW = np.zeros(N)
    for i in range(N):
        chi = gfactors[i]**2*muB**2*spin_values[i]*(spin_values[i]+1)/(3.*kB)
        CW[i] = _NA * _eV * 1E3  * chi 
    return CW.sum()
   