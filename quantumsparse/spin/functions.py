from quantumsparse.constants import muB,g
import numpy as np
from typing import Tuple, List
from quantumsparse.operator import Operator, OpArr
from quantumsparse.bookkeeping import ImplErr
from quantumsparse.matrix import Matrix

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

def euler_to_axis_angle(phi: float, theta: float, psi: float) -> Tuple[float, np.ndarray]:
    """
    Convert Euler angles (phi, theta, psi) in ZYZ convention to axis-angle representation.
    
    Args:
        phi (float): Rotation angle around X-axis (1st rotation)
        theta (float): Rotation angle around Y-axis (2nd rotation)
        psi (float): Rotation angle around Z-axis (3rd rotation)
    
    Returns:
        theta_angle (float): Rotation angle θ
        n (np.ndarray): Rotation axis (unit vector of shape (3,))
    """
    # Use existing Rx, Ry, Rz functions (assume already defined)
    R = Rz(psi) @ Ry(theta) @ Rx(phi)
    R = np.asarray(R)  # convert from np.matrix to ndarray

    # Compute the rotation angle θ from the trace
    trace = np.trace(R)
    theta_angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

    # Compute the rotation axis n
    if np.isclose(theta_angle, 0):
        # Identity rotation — arbitrary axis
        n = np.array([1.0, 0.0, 0.0])
    elif np.isclose(theta_angle, np.pi):
        # Special case: extract axis from (R + I)/2
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diag(A), 0))
        axis /= np.linalg.norm(axis)
        n = axis
    else:
        # General case
        n = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta_angle))
        n /= np.linalg.norm(n)

    return theta_angle, n

def get_Euler_angles(N:int):
    EulerAngles = np.zeros((N, 3))
    EulerAngles[:, 2] = np.linspace(0, 360, N, endpoint=False)
    EulerAngles = np.pi * EulerAngles / 180
    return EulerAngles

def get_unitary_rotation_matrix(spins:Tuple[OpArr,OpArr,OpArr],EulerAngles:np.ndarray)->Tuple[List[Matrix],List[Matrix]]:
    Sx, Sy, Sz = spins
    N = len(Sx)
    U:List[Matrix]  = [None]*N # np.full(N,1.,dtype=object)
    Ud:List[Matrix] = [None]*N # np.full(N,1.,dtype=object)
    for n in range(N):
        phi, theta, psi = EulerAngles[n,:]
        # theta,n = euler_to_axis_angle(phi, theta, psi)
        U[n] = (Sx[n].exp(1.j*phi) @ Sy[n].exp(1.j*theta) @ Sz[n].exp(1.j*psi)).clean()
        Ud[n] = U[n].dagger()
    return U, Ud
    
def cylindrical_coordinates(spins:Tuple[OpArr,OpArr,OpArr])->Matrix:
    from quantumsparse.tools.mathematics import product
    Sx, Sy, Sz = spins
    N = len(Sx)
    EulerAngles = np.zeros((N, 3))
    EulerAngles[:, 2] = np.linspace(0, 360, N, endpoint=False)
    EulerAngles = np.pi * EulerAngles / 180
    U, _ = get_unitary_rotation_matrix(spins, EulerAngles)
    Utot = product(U).clean()
    return Utot

def rotate_spins(spins:Tuple[OpArr,OpArr,OpArr],EulerAngles:np.ndarray,method:str="R")->Tuple[List[Matrix],List[Matrix],List[Matrix]]:
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
        from quantumsparse.matrix import Matrix
        # or annotate explicitly:
        U: List[Matrix]
        Ud: List[Matrix]
        U, Ud = get_unitary_rotation_matrix(spins, EulerAngles)
        for n in range(N):
            SxR[n] = U[n]@Sx[n]@Ud[n]
            SyR[n] = U[n]@Sy[n]@Ud[n]
            SzR[n] = U[n]@Sz[n]@Ud[n]
        return SxR, SyR, SzR
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