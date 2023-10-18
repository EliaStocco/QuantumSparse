# some functions ...
import numpy as np
# from .physical_constants import muB,g

def prepare_opts(opts):
    opts = {} if opts is None else opts
    opts["print"]       = None   if "print"       not in opts else opts["print"]
    opts["sort"]        = True   if "sort"       not in opts else opts["sort"]
    opts["check-low-T"] = 0  if "check-low-T" not in opts else opts["check-low-T"]
    opts["inplace"]     = True   if "inplace"       not in opts else opts["inplace"]
    #opts["return-S"] = False if "return-S" not in opts else opts["return-S"]
    return opts

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

def rotate(EulerAngles,Sx,Sy,Sz):
    N = len(Sx)
    SxR,SyR,SzR = Sx.copy(),Sy.copy(),Sz.copy()
    for n in range(N):
        phi,theta,psi   = EulerAngles[n,:]
        R = Rz(psi) @ Ry(theta) @ Rx(phi)
        temp = R @ np.asarray([Sx[n],Sy[n],Sz[n]])
        SxR[n],SyR[n], SzR[n] = temp[0,0],temp[0,1],  temp[0,2]
    return SxR,SyR,SzR
#
# def magnetic_moment_operator(Sx,Sy,Sz,opts=None):
#     Mx,My,Mz = 0,0,0
#     for sx,sy,sz in zip(Sx,Sy,Sz):
#         Mx += g*muB*sx
#         My += g*muB*sy
#         Mz += g*muB*sz    
#     return Mx,My,Mz

def spherical_coordinates(r,theta,phi,cos=np.cos,sin=np.sin):
    x = r*cos(phi)*sin(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(theta)
    return x,y,z