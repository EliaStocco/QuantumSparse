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