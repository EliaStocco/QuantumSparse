import numpy as np
import pandas as pd
import warnings
from scipy.special import logsumexp
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
    if np.any(T < 0.0):
        raise ValueError("Temperature values must be non-negative.")
    if np.any(np.isclose(T, 0.0)):
        raise ValueError("For null temperatures, please use the function 'statistical_weigths'.")
    return 1.0/(kB*T)

def statistical_weights(T: np.ndarray, E: np.ndarray, tol=TOLERANCE):
    """
    Returns normalized weights w and partition function Z using log-sum-exp for numerical stability.
    Handles T=0 by assigning equal weight to ground states.
    """
    assert E.ndim == 1
    ii = np.isclose(T, 0.0)
    temp = T.copy()
    temp[ii] = 1.0  # temporary safe value
    beta = 1.0 / temp  # or T2beta(temp) if you have it

    Emin = E.min()
    log_w = -np.tensordot(beta, E - Emin, axes=0)  # shape (len(T), len(E))
    
    # compute log-sum-exp for each temperature row
    log_Z = logsumexp(log_w, axis=1, keepdims=True)
    
    # normalized weights
    w = np.exp(log_w - log_Z)

    # handle T=0 separately (equal probability over degenerate GS)
    if np.any(ii):
        deg_gs = np.abs(E - Emin) < tol
        jj_gs = np.where(deg_gs)[0]
        deg = deg_gs.sum()
        for idx in np.where(ii)[0]:
            w[idx, :] = 0.0
            w[idx, jj_gs] = 1.0 / deg

    # unnormalized partition function (including shift by Emin)
    Z = np.exp(log_Z.flatten()) * np.exp(-beta * Emin)  # true Z

    # ensure normalization
    w /= w.sum(axis=1, keepdims=True)
    assert np.allclose(w.sum(axis=1), 1.0)

    return w, Z

def classical_thermal_average_value(T: np.ndarray, E: np.ndarray, Obs: np.ndarray) -> np.ndarray:
    """
    Numerically stable classical thermal average <Obs>_T.
    """
    w, _ = statistical_weights(T=T, E=E)
    return weights2thermal_average(w, Obs)

def correlation_function(
    T: np.ndarray, 
    E: np.ndarray, 
    OpA: Operator,  # or Operator diagonal in eigenbasis
    Psi: np.ndarray,  # eigenvectors
    OpB: Optional[Operator] = None
) -> np.ndarray:
    """
    Numerically stable correlation function <(A-<A>)(B-<B>)>_T for quantum or classical systems.

    Parameters
    ----------
    T : np.ndarray
        Temperatures.
    E : np.ndarray
        Eigenvalues of the Hamiltonian.
    OpA : array-like
        Diagonal elements of operator A in eigenbasis.
    Psi : array-like
        Eigenvectors (unused if OpA already diagonal)
    OpB : array-like, optional
        Diagonal elements of operator B (default: OpB = OpA)
    
    Returns
    -------
    np.ndarray
        Correlation function at each temperature.
    """
    w, _ = statistical_weights(T=T, E=E)
    
    A = expectation_value(OpA, Psi)
    if OpB is None:
        B = A.copy()

    # Compute thermal averages    
    meanA = weights2thermal_average(w, A)
    meanB = weights2thermal_average(w, B)

    # Centered correlation: (A-<A>)(B-<B>)
    centered = (A[None, :] - meanA[:, None]) * (B[None, :] - meanB[:, None])
    corr = np.sum(w * centered, axis=1)

    # Handle tiny negatives due to numerical noise for variance
    if OpB is None:
        if np.any(corr.real < 0):
            warnings.warn(
                "Small negative variance due to floating-point noise. Clipping to zero.",
                RuntimeWarning
            )
        corr = np.maximum(corr.real, 0.0)

    return corr

def quantum_thermal_average_value(T: np.ndarray, E: np.ndarray, Op: Operator, Psi: Matrix) -> np.ndarray:
    """
    Quantum thermal average <Op>_T using eigenbasis Psi.
    """
    # Diagonal elements in eigenbasis
    exp_val = expectation_value(Op, Psi)  # shape (N_states,)
    
    # Classical thermal average over eigenstates
    return classical_thermal_average_value(T, E, exp_val)

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

def weights2thermal_average(w: np.ndarray,Obs: np.ndarray)->np.ndarray:
    assert w.ndim == 2, "error"
    assert Obs.ndim == 1, "error"
    out = np.einsum("ij,j->i",w,Obs)
    return out

def dfT2correlation_function(T: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    w, _ = statistical_weights(T=T, E=df["eigenvalues"].to_numpy())
    A = df["A"].to_numpy()
    B = df["B"].to_numpy() if "B" in df else A
    
    meanA = weights2thermal_average(w, A)
    meanB = weights2thermal_average(w, B)
    
    centered = (A[None, :] - meanA[:, None]) * (B[None, :] - meanB[:, None])
    corr = np.sum(w * centered, axis=1)
    
    if np.allclose(A, B):
        # Handle tiny negative values due to numerical noise
        if np.any(corr.real < 0):
            warnings.warn(
                "Small negative variance encountered due to floating-point noise. Clipping to 0.",
                RuntimeWarning
            )
        corr = np.maximum(corr.real, 0.0)
    return corr

def dfT2thermal_average_and_fluctuation(T: np.ndarray,df:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
    w, _ = statistical_weights(T=T, E=df["eigenvalues"].to_numpy())
    exp_val_A  = weights2thermal_average(w=w,Obs=df["A"].to_numpy())
    tmp = pd.DataFrame({
        "eigenvalues":df["eigenvalues"],
        "A": df["A"], 
        "B": df["A"], 
        "AB": df["A2"]
    })
    return exp_val_A, dfT2correlation_function(T,tmp)

def dfT2susceptibility(T: np.ndarray,df:pd.DataFrame)->np.ndarray:
    beta  = T2beta(T)
    chi = dfT2correlation_function(T,df)
    assert chi.shape == T.shape, "error"
    return beta * chi * _NA * _eV * 1E3  