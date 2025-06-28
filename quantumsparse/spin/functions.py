from quantumsparse.constants import muB,g
import numpy as np
from typing import Tuple
from quantumsparse.operator import Operator
from quantumsparse.bookkeeping import OpArr, ImplErr

def extract_Sxyz(func):
    def wrapper(spins,*argc,**argv):
        if spins is not None:
            Sx,Sy,Sz = spins.Sx, spins.Sy, spins.Sz
            return func(Sx=Sx,Sy=Sy,Sz=Sz,spins=None,*argc,**argv)
        else :
            return func(spins=None,*argc,**argv)
    return wrapper

def magnetic_moments(Sx=None,Sy=None,Sz=None,opts=None)->Tuple[Operator,Operator,Operator]:
    """Create the magnetic moment operator from the spin operators"""
    Mx,My,Mz = 0,0,0
    for sx,sy,sz in zip(Sx,Sy,Sz):
        Mx += g*muB*sx
        My += g*muB*sy
        Mz += g*muB*sz    
    return Mx,My,Mz

# https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
def Rx(phi):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(phi),-np.sin(phi)],
                   [ 0, np.sin(phi), np.cos(phi)]])  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])  
def Rz(psi):
  return np.matrix([[ np.cos(psi), -np.sin(psi), 0 ],
                   [ np.sin(psi), np.cos(psi) , 0 ],
                   [ 0           , 0            , 1 ]])
  
def get_unitary_rotation_matrix(spins:Tuple[OpArr,OpArr,OpArr],EulerAngles:np.ndarray)->Tuple[OpArr,OpArr,OpArr]:
    Sx, Sy, Sz = spins
    N = len(Sx)
    for n in range(N):
        phi, theta, psi = EulerAngles[n,:]
        Sx[n].exp()
    
    

def rotate_spins(spins:Tuple[OpArr,OpArr,OpArr],EulerAngles:np.ndarray,method:str="R")->Tuple[OpArr,OpArr,OpArr]:
    Sx, Sy, Sz = spins
    N = len(Sx)
    SxR,SyR,SzR = Sx.copy(),Sy.copy(),Sz.copy()
    if method == "R":
        # rotation in cartesian space
        v = np.zeros(3,dtype=object)
        for n in range(N):
            phi,theta,psi   = EulerAngles[n,:]
            R = Rz(psi) @ Ry(theta) @ Rx(phi)
            # v = np.asarray([Sx[n],Sy[n],Sz[n]],dtype=object)
            v[0], v[1], v[2] = Sx[n],Sy[n],Sz[n]
            temp = R @ v
            SxR[n],SyR[n], SzR[n] = temp[0,0], temp[0,1], temp[0,2]
    elif method == "U":
        # rotation in Hilbert space
        raise ImplErr
    else:
        raise ValueError(f"'method' can be only 'R' or 'U', ut you provided '{method}'")
    return SxR,SyR,SzR

def from_S2_to_S(S2:np.ndarray)->np.ndarray:
    """
    Parameters
    ----------
    S2 : float
        eigenvalue of the S2 operator

    Returns
    -------
    S : float
        spin value (integer of semi-integer value) such that S(S+1) = S2
    """
    S = (-1 + np.sqrt(1+ 4*S2))/2.0
    return S