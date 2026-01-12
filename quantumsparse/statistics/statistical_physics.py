# some function recalling statistical physics results
import numpy as np
import pandas as pd
from quantumsparse.constants import kB,g,_NA,_eV,muB
from quantumsparse.tools.quantum_mechanics import expectation_value
from quantumsparse.operator import Operator
from quantumsparse.matrix import Matrix
from typing import Optional, Tuple
from quantumsparse.tools.bookkeeping import TOLERANCE


def T2beta(T:np.ndarray)->np.ndarray:
    """
    Calculates the inverse temperature array from a given temperature array.

    Parameters:
        T (np.ndarray): The temperature array.

    Returns:
        np.ndarray: The inverse temperature array.
    """
    return 1.0/(kB*T)

def statistical_weigths(E: np.ndarray, T: np.ndarray, tol=TOLERANCE) -> Tuple[np.ndarray, np.ndarray]:
    assert E.ndim == 1, "only 1D arrays allowed."
    
    # Handle T = 0 separately
    ii = np.isclose(T, 0.0)

    temp = T.copy()
    temp[ii] = 1.0  # safe value to avoid beta=inf

    beta = T2beta(temp)

    # numerical stabilization by shifting the spectrum
    Emin = E.min()
    w = np.exp(-np.tensordot(beta, E - Emin, axes=0))

    # ground-state degeneracy
    gs = Emin
    deg_gs = np.abs(E - gs) < tol
    jj_gs = np.where(deg_gs)[0]
    jj_es = np.where(~deg_gs)[0]
    deg = deg_gs.sum()

    # T = 0 → equal probability over ground states only
    if np.any(ii):
        w[ii][:, jj_gs] = 1.0 / deg
        w[ii][:, jj_es] = 0.0

    # partition function
    Z = w.sum(axis=1)

    # normalize weights
    return w / Z[:, None], Z


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
    w, Z = statistical_weigths(T,E)
    return w @ Obs


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
    exp_val = expectation_value(Op,Psi)
    return classical_thermal_average_value(T,E,exp_val)

def correlation_function(T: np.ndarray, E: np.ndarray, OpA: Operator, Psi: Matrix, OpB: Optional[Operator]=None) -> np.ndarray:
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

def susceptibility(T: np.ndarray,H:Operator,OpA:Operator,OpB:Operator=None)->np.ndarray:
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
    beta  = T2beta(T)
    Chi = correlation_function(T=T,E=H.eigenvalues-np.min(H.eigenvalues),OpA=OpA,Psi=H.eigenstates,OpB=OpB) 
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
   
def dfT2correlation_function(T: np.ndarray,df:pd.DataFrame)->np.ndarray:
    exp_val_A  = classical_thermal_average_value(T,df["eigenvalues"].to_numpy(),df["A"].to_numpy())
    exp_val_B  = classical_thermal_average_value(T,df["eigenvalues"].to_numpy(),df["B"].to_numpy())
    exp_val_AB = classical_thermal_average_value(T,df["eigenvalues"].to_numpy(),df["AB"].to_numpy())
    return exp_val_AB - exp_val_A * exp_val_B

def dfT2susceptibility(T: np.ndarray,df:pd.DataFrame)->np.ndarray:
    beta  = T2beta(T)
    chi = dfT2correlation_function(T,df)
    return beta * chi * _NA * _eV * 1E3  