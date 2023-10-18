# the most adopted interactions in Spin Hamiltonians
import numpy as np
# from ..tools.functions import magnetic_moment_operator

__all__ = [ #"Row_by_Col_mult",\
            "Ising",\
            "Heisenberg",\
            "DM",\
            "anisotropy",\
            "rhombicity",\
            "Zeeman"]

def extract_Sxyz(func):
    def wrapper(spins,*argc,**argv):
        if spins is not None:
            Sx,Sy,Sz = spins.Sx, spins.Sy, spins.Sz
            return func(Sx=Sx,Sy=Sy,Sz=Sz,spins=None,*argc,**argv)
        else :
            return func(spins=None,*argc,**argv)
    return wrapper

    

# def Row_by_Col_mult(A,B,opts=None):
#     """
#     Row by Columns multiplication
#     """
#     if opts is None :
#         opts = {}
#     if "sympy" in opts and opts["sympy"] == True :
#         opts["function"] = lambda a,b : a*b
#     elif "function" not in opts:
#         opts["function"] = lambda a,b : a@b
        
#     return opts["function"](A,B)
        

def Ising(S,couplings=1.0,nn=1,opts=None):
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

@extract_Sxyz
def Heisenberg(Sx=None,Sy=None,Sz=None,spins=None,couplings=1.0,nn=1,opts=None):
    if spins is not None:
        Sx,Sy,Sz = spins.Sx, spins.Sy, spins.Sz
    N = len(Sx)
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
    
    return Ising(Sx,Js[:,0],nn,opts=opts) +\
           Ising(Sy,Js[:,1],nn,opts=opts) +\
           Ising(Sz,Js[:,2],nn,opts=opts)

@extract_Sxyz
def DM(Sx=None,Sy=None,Sz=None,spins=None,couplings=1.0,nn=1,opts=None):
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

def anisotropy(Sz,couplings,opts=None):
    return Ising(Sz,couplings,nn=0,opts=opts)

def rhombicity(Sx,Sy,couplings,opts=None):
    return Ising(Sx,couplings,nn=0,opts=opts) - Ising(Sy,couplings,nn=0,opts=opts)


# def Zeeman(Sx,Sy,Sz,B,opts=None):
#     B = np.asarray(B)
#     Mx,My,Mz = magnetic_moment_operator(Sx,Sy,Sz,opts)    
#     return - ( Mx*B[0] + My*B[1] + Mz*B[2] )
