# the core of QuantumSparse code: a module defining spin operators via Kronecker (tensor) product
import numpy as np
from QuantumSparse.operator import Operator
from QuantumSparse.tools.functions import prepare_opts


class SpinOperators():
    
    def __init__(self,spin_values=None,N=1,S=0.5,opts=None,**argv):
        """
        Parameters
        ----------
        N : int, optional
            number of spin sites
        S : float, optional
            (site-independent) spin value: it must be integer of semi-integer
        spin_values : numpy.array, optional
            numpy.array containing the spin values for each site: they can be integer of semi-integer
            
        Returns
        -------
        None
        """

        # https://realpython.com/python-super/
        # super().__init__(**argv)
        
        # print("\n\tconstructor of \"SpinSystem\" class")     
        opts = prepare_opts(opts)
        if spin_values is not None:
            self.spin_values = spin_values
        else :
            self.spin_values = np.full(N,S)

        self.degeneracies = (2*self.spin_values+1).astype(int)
     
        check = [ not (i%1) for i in self.spin_values*2 ]
        if not np.all(check):
            print("\n\terror: not all spin values are integer or semi-integer: ",self.spin_values)
            raise() 
            
        Sx,Sy,Sz,Sp,Sm = self.compute_spin_operators(self.spin_values,opts)
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz  
        self.Sp = Sp 
        self.Sm = Sm 

        self.basis = self.compute_basis()

    def compute_basis(self):

        from itertools import product
        import pandas as pd

        Nsites = np.arange(len(self.Sz))
        index = np.arange(int(self.Sz[0].shape[0]))
        basis = pd.DataFrame(columns=Nsites,index=index)
        m = [None]*len(Nsites)
        for n in Nsites:
            m[n] = np.linspace(self.spin_values[n],-self.spin_values[n],self.degeneracies[n]) #[ j for j in range(-self.spin_values[n],self.spin_values[n]+1) ]

        tmp = list(product(*m))

        #k = 0
        for i in range(len(tmp)):
            for n in Nsites:
                basis.at[i,n] = tmp[i][n]
                #k += 1

        return basis

   
    @staticmethod
    # @output(operator)
    def compute_spin_operators(spin_values,opts=None):
        """
        Parameters
        ----------
        spin_values : numpy.array
            numpy.array containing the spin values for each site: 
            they can be integer of semi-integer
        opts : dict, optional
            dict containing different options

        Returns
        -------
        Sx : np.array of scipy.sparse
            array of the Sx operators for each spin site, 
            represented with sparse matrices,
            acting on the system (all sites) Hilbert space
        Sy : np.array of scipy.sparse
            array of the Sy operators for each spin site, 
            represented with sparse matrices,
            acting on the system (all sites) Hilbert space
        Sz : np.array of scipy.sparse
            array of the Sz operators for each spin site, 
            represented with sparse matrices,
            acting on the system (all sites) Hilbert space
        """
        
        opts = prepare_opts(opts)
        spin_values = np.asarray(spin_values)    
        from_list_to_str = lambda x :  '[ '+ ' '.join([ "{:d} ,".format(int(i)) if i.is_integer() 
                                                        else "{:f} ,".format(i) 
                                                        for i in x ])[0:-1]+' ]'
            
        print("\n\tcomputing the spin operators")
        print("\t\tinput parameters:")
        NSpin        = len(spin_values)     
        print("\t\t{:>20s} : {:<60d}".format("N spins",NSpin))
        print("\t\t{:>20s} : {:<60s}".format("spin values",from_list_to_str(spin_values)))
        dimensions = SpinOperators.dimensions(spin_values)#(2*spin_values+1).astype(int)
        print("\t\t{:>20s} : {:<60s}".format("dimensions",from_list_to_str(dimensions)))
       
        # print("\t\tallocating single Sz,S+,S- operators (on the single-spin Hilbert space) ... ",end="")
        sz,sp,sm = SpinOperators.single_Szpm(spin_values)
        # print("done")    
        
        # print("\t\tallocating the Sx,Sy,Sz operators (on the system Hilbert space) ... ",end="")  
        Sx,Sy,Sz,Sp,Sm = SpinOperators.system_Sxypm_operators(dimensions,sz,sp,sm)
        # print("done")    

        # for n in range(len(spin_values)):
        #     Sx[n],Sy[n],Sz[n] = operator(Sx[n]), operator(Sy[n]), operator(Sz[n])
               
        return Sx,Sy,Sz,Sp,Sm
    
   
    @staticmethod
    def dimensions(spin_values):
        """
        Parameters
        ----------
        spin_values : numpy.array
            numpy.array containing the spin values for each site: 
                they can be integer of semi-integer

        Returns
        -------
        deg : numpy.array
            array of int representing the Hilbert space dimension for each site
        """
        deg = (2*spin_values+1).astype(int) 
        return deg
    
   
    # @staticmethod
    # @output(operator)
    def compute_S2(self,opts=None):
        """        
        Parameters
        ----------
        Sx : scipy.sparse
            Sx operators of a single spin site, represented with a sparse matrice.
            This operator can be computed by the function "compute_spin_operators"
        Sy : scipy.sparse
            Sy operators of a single spin site, represented with a sparse matrice.
            This operator can be computed by the function "compute_spin_operators"
        Sz : scipy.sparse
            Sz operators of a single spin site, represented with a sparse matrice.
            This operator can be computed by the function "compute_spin_operators"
        opts : dict, optional
            dict containing different options

        Returns
        -------
        S2 : scipy.sparse
            spin square operator 
        """
        Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
        S2 = np.zeros(len(Sx),dtype=object)
        for n,(x,y,z) in enumerate(zip(Sx,Sy,Sz)):
            S2[n] = x@x + y@y + z@z
        return S2
    
   
    # @staticmethod
    # @output(operator)
    def compute_total_S2(self,opts=None)->Operator:
        """        
        Parameters
        ----------
        Sx : np.array of scipy.sparse
            array of the Sx operators for each spin site, represented with sparse matrices.
            This array of operators can be computed by the function "compute_spin_operators"
        Sy : np.array of scipy.sparse
            array of the Sy operators for each spin site, represented with sparse matrices.
            This array of operators can be computed by the function "compute_spin_operators"
        Sz : np.array of scipy.sparse
            array of the Sz operators for each spin site, represented with sparse matrices.
            This array of operators can be computed by the function "compute_spin_operators"
        opts : dict, optional
            dict containing different options

        Returns
        -------
        S2 : scipy.sparse
            total spin square operator 
        """
        Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
        SxTot= Sx.sum()
        SyTot= Sy.sum()
        SzTot= Sz.sum()
        S2 = SxTot@SxTot +   SyTot@SyTot +  SzTot@SzTot
        return S2
            
   
    @staticmethod
    def from_S2_to_S(S2):
        """
        Parameters
        ----------
        S2 : float
            eigenvalue the S2 operator

        Returns
        -------
        S : float
            spin value (integer of semi-integer value) such that S(S+1) = S2
        """
        S = (-1 + np.sqrt(1+ 4*S2))/2.0
        return S

   
    @staticmethod
    # @output(operator)
    def compute_sx(p,m):
        """
        Parameters
        ----------
        p : scipy.sparse
            spin raising operator S+ = Sx + i Sy
        m : scipy.sparse
            spin lowering operator S- = Sx - i Sy

        Returns
        -------
        Sx : scipy.sparse
            Sx operator, computed fromthe inversion of the S+ and S- expressions
        """
        Sx = 1.0/2.0*(p+m)
        return Sx

   
    @staticmethod
    # @output(operator)
    def compute_sy(p,m):
        """
        Parameters
        ----------
        p : scipy.sparse
            spin raising operator S+ = Sx + i Sy
        m : scipy.sparse
            spin lowering operator S- = Sx - i Sy

        Returns
        -------
        Sy : scipy.sparse
            Sy operator, computed fromthe inversion of the S+ and S- expressions
        """
        Sy = -1.j/2.0*(p-m) 
        return Sy
    
   
    @staticmethod
    # @output(operator)
    def single_Szpm(spin_values):
        """
        Parameters
        ----------
        spin_values : numpy.array
            numpy.array containing the spin values for each site: they can be integer of semi-integer

        Returns
        -------
        Sz : numpy.array of scipy.sparse
            array of Sz operators for each site,
            acting on the local (only one site) Hilbert space
        Sp : numpy.array of scipy.sparse
            array of the raising S+ operators for each site,
            acting on the local (only one site) Hilbert space
        Sm : numpy.array of scipy.sparse
            array of lowering Sz operators for each site,
            acting on the local (only one site) Hilbert space
        """
        NSpin = len(spin_values)
        Sz = np.zeros(NSpin,dtype=object) # s z
        Sp = np.zeros(NSpin,dtype=object) # s plus
        Sm = np.zeros(NSpin,dtype=object) # s minus
        dimensions = SpinOperators.dimensions(spin_values)  
        
        for i,s,deg in zip(range(NSpin),spin_values,dimensions):
            
            # print("\t\t",i+1,"/",NSpin,end="\r")
            
            m = np.linspace(s,-s,deg)
            Sz[i] = Operator.diags(m,dtype=float)          
            
            vp = np.sqrt( (s-m)*(s+m+1) )[1:]
            vm = np.sqrt( (s+m)*(s-m+1) )[0:-1]
            Sp[i] = Operator.diags(vp,offsets= 1)
            Sm[i] = Operator.diags(vm,offsets=-1)
    
        return Sz,Sp,Sm

   
    @staticmethod
    # @output(operator)
    def system_Sxypm_operators(dimensions,sz,sp,sm):
        """
        Parameters
        ----------
        dimensions : numpy.array
            numpy.array of integer numbers representing the Hilbert space dimension of each site 
        sz : numpy.array of scipy.sparse
            array of Sz operators for each site,
            acting on the local (only one site) Hilbert space
        sp : numpy.array of scipy.sparse
            array of the raising S+ operators for each site,
            acting on the local (only one site) Hilbert space
        sm : numpy.array of scipy.sparse
            array of lowering S- operators for each site,
            acting on the local (only one site) Hilbert space

        Returns
        -------
        Sx : numpy.array of scipy.sparse
            array of Sx operators for each site,
            acting on the system Hilbert space
        Sy : numpy.array of scipy.sparse
            array of Sy operators for each site,
            acting on the system Hilbert space
        Sz : numpy.array of scipy.sparse
            array of Sz operators for each site,
            acting on the system Hilbert space
        """
        NSpin= len(sz)
        if NSpin != len(sp) or NSpin != len(sm) or NSpin != len(dimensions):
            print("\t\terror in \"compute_Sxy_operators\" function: arrays of different lenght")
            raise()
        Sz = np.zeros(NSpin,dtype=object) # S z
        Sx = np.zeros(NSpin,dtype=object) # S x
        Sy = np.zeros(NSpin,dtype=object) # S y
        Sp = np.zeros(NSpin,dtype=object) # S y
        Sm = np.zeros(NSpin,dtype=object) # S y
        iden = Operator.identity(dimensions)
        
        for zpm,out in zip([sz,sp,sm],[Sz,Sp,Sm]):
            for i in range(NSpin):
                Ops = iden.copy()
                Ops[i] = zpm[i]
                out[i] = Ops[0]
                for j in range(1,NSpin):
                    out[i] = Operator.kron(out[i],Ops[j]) 
                    
        for i in range(NSpin):
            Sx[i] = SpinOperators.compute_sx(Sp[i],Sm[i])
            Sy[i] = SpinOperators.compute_sy(Sp[i],Sm[i])
            
        return Sx,Sy,Sz,Sp,Sm       

    def empty(self):
        return self.Sx[0].empty()    
    
    def identity(self):
        return self.Sx[0].identity(len(self.Sx[0]))
        
    
        # for i in range(NSpin):
    
        #     print("\t",i+1,"/",NSpin,end="\r")
            
        #     if i!=0: #i!=0
        #         mz = iden[0].copy() # matrix z
        #         mp = iden[0].copy() # matrix plus
        #         mm = iden[0].copy() # matrix minus
                
        #         for j in range(1,i):
        #             mz = sparse.kron(mz,iden[j])
        #             mp = sparse.kron(mp,iden[j])
        #             mm = sparse.kron(mm,iden[j])
                    
        #         mz = sparse.kron(mz,sz[i])
        #         mp = sparse.kron(mp,sp[i])
        #         mm = sparse.kron(mm,sm[i])
                
        #         for j in range(i+1,NSpin):
        #             mz = sparse.kron(mz,iden[j])
        #             mp = sparse.kron(mp,iden[j])
        #             mm = sparse.kron(mm,iden[j])
                
        #     else : #i==0    
            
        #         mz = sz[0].copy()
        #         mp = sp[0].copy()
        #         mm = sm[0].copy()      
                
        #         for j in range(1,NSpin):
        #             mz = sparse.kron(mz,iden[j])
        #             mp = sparse.kron(mp,iden[j])
        #             mm = sparse.kron(mm,iden[j])
        #     #
        #     mx = spin_operators.compute_sx(mp,mm)
        #     my = spin_operators.compute_sy(mp,mm)   
    
        #     Sz[i] = mz.copy()       
        #     Sx[i] = mx.copy()
        #     Sy[i] = my.copy()
        
        return Sx,Sy,Sz
