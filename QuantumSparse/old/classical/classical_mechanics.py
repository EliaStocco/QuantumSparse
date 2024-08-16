# symbolic computation through the sympy module to study classical Spin Hamiltonians
import numpy as np
import sympy
from sympy import symbols,cos,sin,Matrix,hessian,lambdify
from scipy.optimize import minimize
from ..functions import spherical_coordinates,prepare_opts

__all__ = [ "get_spin_symbols",\
            "get_angles_symbols",\
            "get_spin_symbols_angles_dependent",\
            "gradient",\
            "get_lambdified_E_grad_Hess",\
            "minimize_classical_energy"]

def get_spin_symbols(NSpin,opts=None):
    opts = prepare_opts(opts)
    print(opts,"\n\t\"get_spin_symbols\" function",file=opts["print"])
    Sx = [sympy.symbols('S^x_%d' % i) for i in range(1,NSpin+1)]
    Sy = [sympy.symbols('S^y_%d' % i) for i in range(1,NSpin+1)]
    Sz = [sympy.symbols('S^z_%d' % i) for i in range(1,NSpin+1)]
    return Sx,Sy,Sz

def get_angles_symbols(NSpin,opts=None):
    opts = prepare_opts(opts)
    print("\n\t\"get_angles_symbols\" function",file=opts["print"])
    print("\n\t\tallocating sympy symbols for angles defining the spins orientations",file=opts["print"])    
    theta = [sympy.symbols('\\theta_%d' % i) for i in range(1,NSpin+1)] # [0,pi]    
    phi   = [sympy.symbols('\\phi_%d'   % i) for i in range(1,NSpin+1)] # [0,2pi)
    return theta,phi

def get_spin_symbols_angles_dependent(theta,phi,spin_values,opts=None):
    opts = prepare_opts(opts)
    print("\n\t\"get_spin_symbols_angles_dependent\" function",file=opts["print"])
    NSpin = len(spin_values)
    Sx,Sy,Sz =  [np.zeros(NSpin,dtype=object)]*3
    print("\n\t\tcomputing cartesian spins through \"spherical_coordinates\" function",file=opts["print"])
    for i in range(NSpin):
        Sx[i],Sy[i],Sz[i] = spherical_coordinates(spin_values[i],theta[i],phi[i],cos=sympy.cos,sin=sympy.sin)
    return list(Sx),list(Sy),list(Sz)

def gradient(f, v): #https://stackoverflow.com/questions/60164477/define-gradient-and-hessian-function-in-python/60165226#60165226
    return Matrix([f]).jacobian(v)

def get_lambdified_E_grad_Hess(H,theta,phi,opts=None):
    opts = prepare_opts(opts)
    print("\n\t\"get_lambdified_E_grad_Hess\" function",file=opts["print"])
    print("\n\t\tcomputing gradient and hessian analytical expression through symbolic computation",file=opts["print"])
    grad_ = gradient(H, theta+phi)
    hess_ = hessian(H, theta+phi)
    print("\n\t\t\"lambifying\" Hamiltonian, gradient and hessian expressions",file=opts["print"])
    energy = lambdify([theta+phi],H,'numpy')
    grad   = lambdify([theta+phi],grad_,'numpy')
    hess   = lambdify([theta+phi],hess_,'numpy')
    return energy,grad,hess

def minimize_classical_energy(H,NSpin,theta,phi,method='L-BFGS-B',tol=1e-6,NIterMax=10,opts=None):
    opts = prepare_opts(opts)
    print("\n\t\"minimize_classical_energy\" function",file=opts["print"])
    print("\n\t\t{:>15s}:{:<10s}".format("method",method),file=opts["print"])
    print("\n\t\t{:>15s}:{:<10.1e}".format("tol",tol),file=opts["print"])
    print("\n\t\t{:>15s}:{:<10d}".format("max n. iter.",NIterMax),file=opts["print"])
    
    energy,grad,hess = get_lambdified_E_grad_Hess(H,theta,phi)
    jac    = lambda x : grad(x).T.flatten()
   
    bounds = [(0,np.pi) for i in range(NSpin)] + [(0,2*np.pi) for i in range(NSpin)]
    E     = np.zeros(NIterMax)
    angles = np.zeros((NIterMax,NSpin*2))
    print("\n\t\tstarting cycle of minimizations",file=opts["print"])
    for n in range(NIterMax):
        string = "%d/%d"%(n+1,NIterMax) ; print(string,file=opts["print"])
        x0 = np.random.rand(NSpin*2)*np.asarray( [ np.pi for i in range(NSpin)] + [ 2*np.pi for i in range(NSpin)] )
        res = minimize(fun=energy,x0=x0,jac=jac,hess=hess,bounds=bounds,method=method,tol=tol)
        #
        E[n] = res.fun
        angles[n,:] = res.x    
        #string = "energy: %f"   %(res.fun)                   ; print(string,file=opts["print"])
        #string = " theta: %f"   %(res.x [0:NSpin]*180/np.pi) ; print(string,file=opts["print"])
        #string = "   phi: %f\n" %(res.x [NSpin: ]*180/np.pi) ; print(string,file=opts["print"])
    print("\n\t\tfinished cycles of minimizations",file=opts["print"])
    if "return-all" not in opts or opts["return-all"] == True :
        return {"lowest-energy-index":np.argmin(E),"energy":E,"angles":angles,"Hamiltonian":energy,"gradient":grad,"hessian":hess}
    else :
        return {"energy":np.min(E),"angles":angles[np.argmin(E)],"Hamiltonian":energy,"gradient":grad,"hessian":hess}
    