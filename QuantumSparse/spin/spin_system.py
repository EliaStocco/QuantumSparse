# "spin_system" class
import numpy as np
from .spin_operators import spin_operators 
from ..constants.constants import muB,g
from ..system.system import system


class spin_system(spin_operators,system):
    
   
    def __init__(self,N=1,S=0.5,spin_values=None,classical=False,opts=None,*args,**kwargs):
        
        self.classical = classical     
        kwargs["N"] = N
        kwargs["S"] = S
        kwargs["spin_values"] = spin_values
        kwargs["opts"] = opts
        kwargs["H"] = 0 
        #super().__init__(N=N,S=S,spin_values=spin_values,opts=opts,*args,**kwargs)
        super().__init__(*args,**kwargs)
        #super(system).__init__(H=0,args=args,kwargs=kwargs)
        
        # https://realpython.com/python-super/
        #super().__init__(*args, **kwargs)
        
        return
    
   
    def nsites(self):
        N = len(self.spin_values)
        return N
    
   
    @staticmethod
    def Row_by_Col_mult(A,B,opts=None):
        """
        Row by Columns multiplication
        """
        if opts is None :
            opts = {}
        if "sympy" in opts and opts["sympy"] == True :
            opts["function"] = lambda a,b : a*b
        elif "function" not in opts:
            opts["function"] = lambda a,b : a@b            
        return opts["function"](A,B)
        
   
    def add_Ising(self,couplings=1.0,nn=1,opts=None):
        opts = {} if opts is None else opts
        opts["sympy"] = True if self.classical else False
        self.Hamiltonian += spin_system.Ising(self.Sz,couplings,nn,opts)
        return 
    
    @staticmethod
    def Ising(Sz,couplings=1.0,nn=1,opts=None):
        H = 0
        N = len(Sz)
        index_I = np.arange(0,N)
        index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
        if hasattr(couplings,'__len__') == False :
            Js = np.full(N,couplings)
        else :
            Js = couplings            
        for i,j,J in zip(index_I,index_J,Js):
            H +=J * spin_system.Row_by_Col_mult(Sz[i],Sz[j],opts=opts)            
        return H
    
   
    def add_Heisenberg(self,couplings=1.0,nn=1,opts=None):
        opts = {} if opts is None else opts
        opts["sympy"] = True if self.classical else False
        self.Hamiltonian += spin_system.Heisenberg(self.Sx,self.Sy,self.Sz,couplings,nn,opts)
        return
    
    @staticmethod
    def Heisenberg(Sx,Sy,Sz,couplings=1.0,nn=1,opts=None):
        N = len(Sx)
        Js = np.asarray(couplings)
        if len(Js.shape) != 2 : 
            Js = np.full((N,3),couplings)            
        H = spin_system.Ising(Sx,Js[:,0],nn,opts=opts) +\
            spin_system.Ising(Sy,Js[:,1],nn,opts=opts) +\
            spin_system.Ising(Sz,Js[:,2],nn,opts=opts)            
        return H 
    
   
    def add_DM(self,couplings=1.0,nn=1,opts=None):
        opts = {} if opts is None else opts
        opts["sympy"] = True if self.classical else False
        self.Hamiltonian += spin_system.DM(self.Sx,self.Sy,self.Sz,couplings,nn,opts)
        return
    
    @staticmethod
    def DM(Sx,Sy,Sz,couplings=1.0,nn=1,opts=None):
        H = 0
        N = len(Sx)
        index_I = np.arange(0,N)
        index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
        Js = np.asarray(couplings)
        if len(Js.shape) != 2 : 
            Js = np.full((N,3),couplings)            
        RbC = lambda a,b : spin_system.Row_by_Col_mult(a,b,opts=opts)            
        for i,j,J in zip(index_I,index_J,Js):
            H += J[0] * ( RbC(Sy[i],Sz[j]) - RbC(Sz[i],Sy[j])) 
            H += J[1] * ( RbC(Sz[i],Sx[j]) - RbC(Sx[i],Sz[j]))
            H += J[2] * ( RbC(Sx[i],Sy[j]) - RbC(Sy[i],Sx[j]))             
        return H
    
   
    def add_anisotropy(self,couplings,opts=None):
        opts = {} if opts is None else opts
        opts["sympy"] = True if self.classical else False
        self.Hamiltonian += spin_system.anisotropy(self.Sz,couplings,opts)
        return
    
    @staticmethod
    def anisotropy(Sz,couplings,opts=None):
        H = spin_system.Ising(Sz,couplings,nn=0,opts=opts)
        return H
    
   
    def add_rhombicity(self,couplings,opts=None):
        opts = {} if opts is None else opts
        opts["sympy"] = True if self.classical else False
        self.Hamiltonian += spin_system.anisotropy(self.Sx,self.Sy,couplings,opts)
        return
    
    @staticmethod
    def rhombicity(Sx,Sy,couplings,opts=None):
        H = spin_system.Ising(Sx,couplings,nn=0,opts=opts) - \
            spin_system.Ising(Sy,couplings,nn=0,opts=opts)
        return H
    
   
    def add_Zeeman(self,B,opts=None):
        opts = {} if opts is None else opts
        opts["sympy"] = True if self.classical else False
        self.Hamiltonian += spin_system.Zeeman(self.Sx,self.Sy,self.Sz,B,opts)
        return
    
    @staticmethod
    def Zeeman(Sx,Sy,Sz,B,opts=None):
        B = np.asarray(B)
        Mx,My,Mz = spin_system.magnetic_moment_operator(Sx,Sy,Sz,opts)   
        H = - ( Mx*B[0] + My*B[1] + Mz*B[2] )
        return H 
    
   
    @staticmethod
    def magnetic_moment_operator(Sx,Sy,Sz,opts=None):
        Mx,My,Mz = 0,0,0
        for sx,sy,sz in zip(Sx,Sy,Sz):
            Mx += g*muB*sx
            My += g*muB*sy
            Mz += g*muB*sz    
        return Mx,My,Mz

   
    def Zeeman_diagram(self,Barr,NLanczos=100,tol=1E-8,MaxDim=100,opts=None):
        opts = {} if opts is None else opts
        opts["inplace"] = False
        
        NN = len(Barr)
        diagram = np.full((NLanczos,NN),np.nan)
        E0 = None
        for n,B in enumerate(Barr):
            print(n+1,"/",NN)
            HB = self.Hamiltonian + spin_system.Zeeman(self.Sx,self.Sy,self.Sz,B)
            E,Psi = self.diagonalize(HB,NLanczos=NLanczos,tol=tol,MaxDim=MaxDim,opts=opts)
            if n == 0 :
                E0 = min(E)
            diagram[:,n] = E-E0
        return diagram #eV               
            