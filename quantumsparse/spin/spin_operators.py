import numpy as np
import pandas as pd
from typing import List
from quantumsparse.operator import Operator, OpArr
from typing import Tuple, Union, TypeVar


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

    def __init__(self:T,spin_values:np.ndarray=None,N:int=1,S:Union[int,float]=0.5,test:bool=False,**argv):
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
        
        if test:
            for n in range(len(self.spin_values)):
                testx = self.Sx[n].test_eigensolution()
                testy = self.Sy[n].test_eigensolution()
                testz = self.Sz[n].test_eigensolution()
                
                assert testx.norm() < 1e-10, f"Spin Sx[{n}] is not diagonalized: {testx.norm()}"
                assert testy.norm() < 1e-10, f"Spin Sy[{n}] is not diagonalized: {testy.norm()}"
                assert testz.norm() < 1e-10, f"Spin Sz[{n}] is not diagonalized: {testz.norm()}"
            
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
    iden:List[Operator] = Operator.identity(dimensions)
    for i in iden:
        i.diagonalize()
    
    for zpm,out in zip([sx,sy,sz,sp,sm],[Sx,Sy,Sz,Sp,Sm]):
        for i in range(NSpin):
            Ops = iden.copy()
            Ops[i] = zpm[i]
            out[i] = Ops[0]
            for j in range(1,NSpin):
                # out[i] = Operator.kron(out[i],Ops[j]) 
                out[i] = out[i].kronecker(Ops[j])
        
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
    Sx:List[Operator] = np.zeros(NSpin,dtype=object) # s x
    Sy:List[Operator] = np.zeros(NSpin,dtype=object) # s y
    Sz:List[Operator] = np.zeros(NSpin,dtype=object) # s z
    Sp:List[Operator] = np.zeros(NSpin,dtype=object) # s plus
    Sm:List[Operator] = np.zeros(NSpin,dtype=object) # s minus
    dimensions = spin2dim(spin_values)  
    
    for i,s,deg in zip(range(NSpin),spin_values,dimensions):
                
        m = np.linspace(s,-s,deg)
        Sz[i] = Operator.diags(m,dtype=float)          
        
        vp = np.sqrt( (s-m)*(s+m+1) )[1:]
        vm = np.sqrt( (s+m)*(s-m+1) )[0:-1]
        Sp[i] = Operator.diags(vp,offsets= 1)
        Sm[i] = Operator.diags(vm,offsets=-1)
        
        Sx[i] = compute_sx(Sp[i],Sm[i])
        Sy[i] = compute_sy(Sp[i],Sm[i])
        
        Sx[i].diagonalize()
        Sy[i].diagonalize()
        Sz[i].diagonalize()

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
    dimensions = spin2dim(spin_values)
    sx,sy,sz,sp,sm = single_Szpm(spin_values)
    Sx,Sy,Sz,Sp,Sm = system_Sxypm_operators(dimensions,sx,sy,sz,sp,sm)
    return Sx,Sy,Sz,Sp,Sm
