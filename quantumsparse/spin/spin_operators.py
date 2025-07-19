import numpy as np
import pandas as pd
from typing import List
from quantumsparse.operator import Operator, OpArr
from quantumsparse.tools.functions import prepare_opts
from typing import Tuple, Union, Any, TypeVar


T = TypeVar('T', bound='SpinOperators')

class SpinOperators:
    """Class to represent a spin system (ideally a chain of spins)."""
    
    spin_values:np.ndarray
    Sx:List[Operator]
    Sy:List[Operator]
    Sz:List[Operator]
    Sp:List[Operator]
    Sm:List[Operator]
    basis:pd.DataFrame

    def __init__(self:T,spin_values:np.ndarray=None,N:int=1,S:Union[int,float]=0.5,**argv):
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
        # opts = prepare_opts(opts)
        if spin_values is not None:
            self.spin_values = spin_values
        else :
            self.spin_values = np.full(N,S)

        # self.degeneracies = (2*self.spin_values+1).astype(int)
        self.degeneracies = spin2dim(self.spin_values)
     
        check = [ not (i%1) for i in self.spin_values*2 ]
        if not np.all(check):
            print("\n\terror: not all spin values are integer or semi-integer: ",self.spin_values)
            raise() 
            
        Sx,Sy,Sz,Sp,Sm = compute_spin_operators(self.spin_values)
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz  
        self.Sp = Sp 
        self.Sm = Sm 
        
        from quantumsparse.spin.functions import rotate_spins, get_unitary_rotation_matrix
        
        for n,Sz in enumerate(self.Sz):
            assert Sz.is_diagonal(), "error: the Sz operator is not diagonal"
            Sz.diagonalize() 
            # test = Sz.test_eigensolution()
            # assert test.norm() < 1e-10, "error: the Sz operator does not have the correct eigenvalues"
            
            euler = np.zeros((len(self.spin_values),3),dtype=float)
            euler[:,1] = np.pi/2. # Sz -> Sx
            test_Sx = get_unitary_rotation_matrix((Sx,Sy,Sz),euler)
            
            pass
            
        for n,Sx in enumerate(self.Sx):
            #assert Sx.is_diagonal(), "error: the Sx operator is not diagonal"
            Sx.diagonalize() 
            # test = Sx.test_eigensolution()
            # assert test.norm() < 1e-10, "error: the Sx operator does not have the correct eigenvalues"
        
        for n,Sy in enumerate(self.Sy):
            # assert Sy.is_diagonal(), "error: the Sy operator is not diagonal"
            Sy.diagonalize() 
            # test = Sy.test_eigensolution()
            # assert test.norm() < 1e-10, "error: the Sy operator does not have the correct eigenvalues"
            
        

        self.basis:pd.DataFrame = self.compute_basis()

    def compute_basis(self)->pd.DataFrame:
        """
        Compute the basis of the Hilbert space of the spin system.
        
        Returns
        -------
        basis : pd.DataFrame
            DataFrame with the basis of the Hilbert space of the spin system.
        """

        from itertools import product
        

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

    def compute_S2(self:T,opts=None)->OpArr:
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

    def empty(self):
        return self.Sx[0].empty()    
    
    def identity(self):
        return self.Sx[0].identity(len(self.Sx[0]))
    
    def check_site_independence(self:T,tolerance:float=1e-10)->bool:
        
        to_run = [["Sx","Sx"],["Sy","Sy"],["Sz","Sz"],["Sx","Sy"],["Sx","Sz"],["Sy","Sz"]]
        
        N = len(self.spin_values)
        for op in to_run:
            OpA = getattr(self,op[0]) 
            OpB = getattr(self,op[1])
            for i in range(N):
                for j in range(i+1,N):
                    comm:Operator = Operator.commutator(OpA[i],OpB[j])
                    norm = comm.norm()
                    assert norm < tolerance, f"The operator {op[0]}[{i}] does not commute with the operator {op[0]}[{j}]: {norm}"
            
def spin2dim(spin_values: np.ndarray)->np.ndarray:
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
    assert spin_values.ndim == 1, "spin_values must be a 1D array"
    deg =  (2*spin_values+1).astype(int) 
    return deg

def dim2spin(dim: np.ndarray)->np.ndarray:
    assert dim.ndim == 1, "spin_values must be a 1D array"
    spin = (dim-1)/2
    return spin.astype(int) 
        
def compute_sx(p:Operator,m:Operator)->Operator:
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

def compute_sy(p:Operator,m:Operator)->Operator:
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
    
def system_Sxypm_operators(dimensions,sx,sy,sz,sp,sm)->Tuple[OpArr,OpArr,OpArr,OpArr,OpArr]:
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
        raise ValueError("Arrays of different length in 'compute_Sxy_operators'")
    Sz = np.zeros(NSpin,dtype=object) # S z
    Sx = np.zeros(NSpin,dtype=object) # S x
    Sy = np.zeros(NSpin,dtype=object) # S y
    Sp = np.zeros(NSpin,dtype=object) # S plus
    Sm = np.zeros(NSpin,dtype=object) # S minus
    iden = Operator.identity(dimensions)
    
    for zpm,out in zip([sx,sy,sz,sp,sm],[Sx,Sy,Sz,Sp,Sm]):
        for i in range(NSpin):
            Ops = iden.copy()
            Ops[i] = zpm[i]
            out[i] = Ops[0]
            for j in range(1,NSpin):
                out[i] = Operator.kron(out[i],Ops[j]) 
                
    # for i in range(NSpin):
    #     Sx[i] = compute_sx(Sp[i],Sm[i])
    #     Sy[i] = compute_sy(Sp[i],Sm[i])
    
    # for i in range(NSpin):
    #     a = compute_sx(Sp[i],Sm[i])
    #     assert (a-Sx[i]).norm() < 1e-10, f"Error in Sx[{i}]"
    #     a = compute_sy(Sp[i],Sm[i])
    #     assert (a-Sy[i]).norm() < 1e-10, f"Error in Sy[{i}]"
        
    return Sx,Sy,Sz,Sp,Sm       

def single_Szpm(spin_values:np.ndarray)->Tuple[OpArr,OpArr,OpArr]:
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
    Sx = np.zeros(NSpin,dtype=object) # s x
    Sy = np.zeros(NSpin,dtype=object) # s y
    Sz = np.zeros(NSpin,dtype=object) # s z
    Sp = np.zeros(NSpin,dtype=object) # s plus
    Sm = np.zeros(NSpin,dtype=object) # s minus
    dimensions = spin2dim(spin_values)  
    
    for i,s,deg in zip(range(NSpin),spin_values,dimensions):
        
        # print("\t\t",i+1,"/",NSpin,end="\r")
        
        m = np.linspace(s,-s,deg)
        Sz[i] = Operator.diags(m,dtype=float)          
        
        vp = np.sqrt( (s-m)*(s+m+1) )[1:]
        vm = np.sqrt( (s+m)*(s-m+1) )[0:-1]
        Sp[i] = Operator.diags(vp,offsets= 1)
        Sm[i] = Operator.diags(vm,offsets=-1)
        
        Sx[i] = compute_sx(Sp[i],Sm[i])
        Sy[i] = compute_sy(Sp[i],Sm[i])

    return Sx,Sy,Sz,Sp,Sm

def compute_spin_operators(spin_values:np.ndarray)->Tuple[OpArr,OpArr,OpArr,OpArr,OpArr]:
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
    
    spin_values = np.asarray(spin_values)    
    # from_list_to_str = lambda x :  '[ '+ ' '.join([ "{:d} ,".format(int(i)) if i.is_integer() 
    #                                                 else "{:f} ,".format(i) 
    #                                                 for i in x ])[0:-1]+' ]'
        
    # print("\n\tcomputing the spin operators")
    # print("\t\tinput parameters:")
    # NSpin        = len(spin_values)     
    # print("\t\t{:>20s} : {:<60d}".format("N spins",NSpin))
    # print("\t\t{:>20s} : {:<60s}".format("spin values",from_list_to_str(spin_values)))
    dimensions = spin2dim(spin_values)#(2*spin_values+1).astype(int)
    # print("\t\t{:>20s} : {:<60s}".format("dimensions",from_list_to_str(dimensions)))
    
    # print("\t\tallocating single Sz,S+,S- operators (on the single-spin Hilbert space) ... ",end="")
    sx,sy,sz,sp,sm = single_Szpm(spin_values)
    # print("done")    
    
    # print("\t\tallocating the Sx,Sy,Sz operators (on the system Hilbert space) ... ",end="")  
    Sx,Sy,Sz,Sp,Sm = system_Sxypm_operators(dimensions,sx,sy,sz,sp,sm)
    
    # from quantumsparse.hilbert import embed_operators
    # testz, testp, testm = embed_operators([sz,sp,sm],dimensions,normalize=False)
    
    # assert np.allclose( [ (testz[i] - Sz[i] ).norm() for i in range(len(testz)) ] , 0 )
    # assert np.allclose( [ (testp[i] - Sp[i] ).norm() for i in range(len(testz)) ] , 0 )
    # assert np.allclose( [ (testm[i] - Sm[i] ).norm() for i in range(len(testz)) ] , 0 )
    # print("done")    

    # for n in range(len(spin_values)):
    #     Sx[n],Sy[n],Sz[n] = operator(Sx[n]), operator(Sy[n]), operator(Sz[n])
            
    return Sx,Sy,Sz,Sp,Sm
