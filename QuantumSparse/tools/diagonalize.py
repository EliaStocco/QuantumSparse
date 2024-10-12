# module to solve the eigenvalue problem: Lanczos or full diag. methods are provided
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
from ..tools.functions import prepare_opts
# from .interactions import Zeeman

# to be modified
def diagonalize(H,NLanczos=100,tol=1E-8,MaxDim=100,opts=None):
    opts = prepare_opts(opts)    
    print("\n\t\"diagonalize\" function")    
    dimension = H.shape[0]   
    print("\t\t{:>40s}\t:{:>6d} x {:<10d}".format("Hamilonian matrix of dimension",dimension,dimension))
    print("\t\t{:>40s}\t:{:>6.2E}".format("Lanczos tolerance",tol))
    print("\t\t{:>40s}\t:{:>6d}".format("Lanczos n. of eigenvalues",NLanczos))
    print("\t\t{:>40s}\t:{:>6d}".format("Apply full diagonalization up to dimension",MaxDim))
    
    print("")
    NLanczos= min ( NLanczos , H.shape[0]-1)
    
    if dimension >= MaxDim :
        print("\t\tusing Lanczos method")
        print("\t\t{:>40s}\t:\t{:<10d}".format("n. eigenvalues in Lanczos method",NLanczos))
        
        if opts["check-low-T"] > 0 :
            print("\t\t------------------------------------------------")
            print("\t\theuristic test on the number of used eigenstates")
            print("\t\tcontrolling that their number is sufficient to describe low-T properties")
            print("\t\tlow-T properties should be well described up to {:>4.2f} K (Kelvin)".format(opts["check-low-T"]))
            Emin,Psi = sparse.linalg.eigsh(H,k=1,tol=tol,which="SA")
            Emax,Psi = sparse.linalg.eigsh(H,k=1,tol=tol,which="LA")
            print("\t\t lowest eigenvalue : {:>6.4f} meV".format(Emin[0]*1000))
            print("\t\thighest eigenvalue : {:>6.4f} meV".format(Emax[0]*1000))
            # 1K = 0.08617328149741 meV
            Nmin = int(dimension * opts["check-low-T"] * 0.08617328149741 *1E-3 / ( Emax[0]- Emin[0]))
            print("\t\tsmallest number of eigenstates to be used : {:>6d}".format(Nmin))
            if Nmin > NLanczos:
                print("\n\t\tWARNING : you should increase NLanczos at least up to %d"%(Nmin))
            else :
                print("\n\t\tthe number of used eigenstates seems to be sufficient")
            print("\t\t------------------------------------------------\n")
            
        E,Psi = sparse.linalg.eigsh(H,k=NLanczos,tol=tol,which="SA")
        # SA : Smallest Algebraic
    else :
        print("\t\tusing a full diagonalization method")
        E,Psi = np.linalg.eigh(H.todense())
        
    # check that eigevectors are orthogonalized: 
    # np.sqrt(np.square(np.absolute(Psi)).sum(axis=0)) = [1,1,1,1...]
        
    # sort
    if opts["sort"] :
        index = np.argsort(E)
        E = E[index]
        Psi = Psi[:,index]
        
    return E,Psi

# def Zeeman_diagram(H,Barr,Sx,Sy,Sz,NLanczos=100,tol=1E-8,MaxDim=100,opts=None):
#     NN = len(Barr)
#     diagram = np.full((NLanczos,NN),np.nan)
#     E0 = None
#     for n,B in enumerate(Barr):
#         print(n+1,"/",NN)
#         HB = H + Zeeman(Sx,Sy,Sz,B)
#         E,Psi = diagonalize_Hamiltonian(HB,NLanczos=NLanczos,tol=tol,MaxDim=MaxDim,opts=opts)
#         if n == 0 :
#             E0 = min(E)
#         diagram[:,n] = E-E0
#     return diagram #eV
