from ..constants.constants import muB,g
import numpy as np

def extract_Sxyz(func):
    def wrapper(spins,*argc,**argv):
        if spins is not None:
            Sx,Sy,Sz = spins.Sx, spins.Sy, spins.Sz
            return func(Sx=Sx,Sy=Sy,Sz=Sz,spins=None,*argc,**argv)
        else :
            return func(spins=None,*argc,**argv)
    return wrapper

def magnetic_moments(Sx=None,Sy=None,Sz=None,opts=None):
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

@extract_Sxyz
def rotate_spins(Sx=None,Sy=None,Sz=None,spins=None,EulerAngles=None):
    N = len(Sx)
    SxR,SyR,SzR = Sx.copy(),Sy.copy(),Sz.copy()
    v = np.zeros(3,dtype=object)
    for n in range(N):
        phi,theta,psi   = EulerAngles[n,:]
        R = Rz(psi) @ Ry(theta) @ Rx(phi)
        # v = np.asarray([Sx[n],Sy[n],Sz[n]],dtype=object)
        v[0], v[1], v[2] = Sx[n],Sy[n],Sz[n]
        temp = R @ v
        SxR[n],SyR[n], SzR[n] = temp[0,0],temp[0,1],  temp[0,2]
    return SxR,SyR,SzR
