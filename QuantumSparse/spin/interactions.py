"""The most adopted interactions in Spin Hamiltonians"""

import numpy as np
from ..operator import Operator
    
def Ising(S:np.ndarray,couplings=1.0,nn=1,opts=None)->Operator:
    """
    This function generates the Ising interaction Hamiltonian.

    Parameters:
    S (np.ndarray): The spin operators.
    couplings (float or np.ndarray): The coupling strengths. Default is 1.0.
    nn (int): The number of nearest neighbors. Default is 1.
    opts (dict): Optional parameters. Default is None.

    Returns:
    Operator: The Ising interaction Hamiltonian.
    """
    H = 0
    N = len(S)
    index_I = np.arange(0,N)
    index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
    if hasattr(couplings,'__len__') == False :
        Js = np.full(N,couplings)
    else :
        Js = couplings
        
    for i,j,J in zip(index_I,index_J,Js):
        #H +=J * Row_by_Col_mult(Ops[i],Ops[j],opts=opts)
        H = H + J * ( S[i] @ S[j] )
        
    return H

def Heisenberg(Sx:np.ndarray=None,Sy:np.ndarray=None,Sz:np.ndarray=None,couplings=1.0,nn=1,opts=None)->Operator:
    """
    This function generates the Heisenberg interaction Hamiltonian.
    
    Parameters:
    Sx (np.ndarray): The x-component of the spin operators. Default is None.
    Sy (np.ndarray): The y-component of the spin operators. Default is None.
    Sz (np.ndarray): The z-component of the spin operators. Default is None.
    couplings (float or np.ndarray): The coupling constants. Default is 1.0.
    nn (int): The number of nearest neighbors. Default is 1.
    opts (dict): Optional parameters. Default is None.
    
    Returns:
    Operator: The Heisenberg interaction Hamiltonian.
    """
    N = len(Sx)
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
    
    return Ising(Sx,Js[:,0],nn,opts=opts) +\
           Ising(Sy,Js[:,1],nn,opts=opts) +\
           Ising(Sz,Js[:,2],nn,opts=opts)


def DM(Sx:np.ndarray=None,Sy:np.ndarray=None,Sz:np.ndarray=None,couplings=1.0,nn=1,opts=None)->Operator:
    """
    This function generates the Dzyaloshinskii-Moriya (DM) interaction Hamiltonian.
    
    Parameters:
    Sx (np.ndarray): The x-component of the spin operators. Default is None.
    Sy (np.ndarray): The y-component of the spin operators. Default is None.
    Sz (np.ndarray): The z-component of the spin operators. Default is None.
    couplings (float or np.ndarray): The coupling strengths. Default is 1.0.
    nn (int): The number of nearest neighbors. Default is 1.
    opts (dict): Optional parameters. Default is None.
    
    Returns:
    Operator: The DM interaction Hamiltonian.
    """
    H = 0
    N = len(Sx)
    index_I = np.arange(0,N)
    index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
        
    #RbC = lambda a,b : Row_by_Col_mult(a,b,opts=opts)
        
    for i,j,J in zip(index_I,index_J,Js):
        # H += J[0] * ( RbC(Sy[i],Sz[j]) - RbC(Sz[i],Sy[j])) 
        # H += J[1] * ( RbC(Sz[i],Sx[j]) - RbC(Sx[i],Sz[j]))
        # H += J[2] * ( RbC(Sx[i],Sy[j]) - RbC(Sy[i],Sx[j])) 
        H += J[0] * ( Sy[i]@Sz[j] - Sz[i]@Sy[j])
        H += J[1] * ( Sz[i]@Sx[j] - Sx[i]@Sz[j])
        H += J[2] * ( Sx[i]@Sy[j] - Sy[i]@Sx[j]) 
        
    return H

def anisotropy(Sz:np.ndarray,couplings=1,opts=None)->Operator:
    """
    This function calculates the anisotropy term in a spin Hamiltonian.

    Parameters:
    Sz (np.ndarray): The z-component of the spin operator.
    couplings: The coupling strength of the anisotropy term.
    opts (dict): Optional parameters. Defaults to None.

    Returns:
    Operator: The anisotropy term in the spin Hamiltonian.
    """
    return Ising(Sz,couplings,nn=0,opts=opts)

def rhombicity(Sx:np.ndarray,Sy:np.ndarray,couplings,opts=None)->Operator:
    """
    This function calculates the rhombicity term in a spin Hamiltonian.
    
    Parameters:
    Sx (np.ndarray): The x-component of the spin operator.
    Sy (np.ndarray): The y-component of the spin operator.
    couplings: The coupling strength of the rhombicity term.
    opts (dict): Optional parameters.
    
    Returns:
    Operator: The rhombicity term in the spin Hamiltonian.
    """
    return Ising(Sx,couplings,nn=0,opts=opts) - Ising(Sy,couplings,nn=0,opts=opts)

def BiQuadraticIsing(S:np.ndarray,couplings=1.0,nn=1,opts=None)->Operator:
    """
    This function generates the biquadratic Ising interaction Hamiltonian.

    Parameters:
    S (np.ndarray): The spin operators.
    couplings (float or np.ndarray): The coupling strengths. Default is 1.0.
    nn (int): The number of nearest neighbors. Default is 1.
    opts (dict): Optional parameters. Default is None.

    Returns:
    Operator: The biquadratic Ising interaction Hamiltonian.
    """
    H = 0
    N = len(S)
    index_I = np.arange(0,N)
    index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
    if not hasattr(couplings,'__len__') :
        Js = np.full(N,couplings)
    else :
        Js = couplings
        
    for i,j,J in zip(index_I,index_J,Js):
        #H +=J * Row_by_Col_mult(Ops[i],Ops[j],opts=opts)
        tmp = S[i] @ S[j]
        H = H + J * ( tmp @ tmp ) # biquadratic
        
    return H

def BiQuadraticHeisenberg(Sx:np.ndarray=None,Sy:np.ndarray=None,Sz:np.ndarray=None,couplings=1.0,nn=1,opts=None)->Operator:
    """
    This function generates the biquadratic Heisenberg interaction Hamiltonian.

    Parameters:
    Sx (np.ndarray): The x-component of the spin operators. Default is None.
    Sy (np.ndarray): The y-component of the spin operators. Default is None.
    Sz (np.ndarray): The z-component of the spin operators. Default is None.
    couplings (float or np.ndarray): The coupling strengths. Default is 1.0.
    nn (int): The number of nearest neighbors. Default is 1.
    opts (dict): Optional parameters. Default is None.

    Returns:
    Operator: The biquadratic Heisenberg interaction Hamiltonian.
    """
    N = len(Sx)
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
    
    return BiQuadraticIsing(Sx,Js[:,0],nn,opts=opts) +\
           BiQuadraticIsing(Sy,Js[:,1],nn,opts=opts) +\
           BiQuadraticIsing(Sz,Js[:,2],nn,opts=opts)