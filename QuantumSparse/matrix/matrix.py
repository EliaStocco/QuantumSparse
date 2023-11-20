from scipy.linalg import eigh, eig
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# from scipy.sparse import issparse, diags, hstack
from copy import copy
from scipy.sparse import bmat
import numpy as np
import numba 
from QuantumSparse.errors import ImplErr
from typing import TypeVar, Union
T = TypeVar('T') 
dtype = csr_matrix
NoJacobi = 8

# class get_class(type):
#     def __new__(cls, name, bases, attrs):
#         # Iterate through the attributes of the class
#         if issubclass(dtype, sparse.spmatrix) :
#             attrs["module"] = sparse
#         else :
#             raise ImplErr
        
#         bases = bases + (dtype,)

#         # Create the class using the modified attributes
#         return super().__new__(cls, name, bases, attrs)
    

# class matrix(metaclass=get_class)
class matrix(csr_matrix):
    """class to handle matrices in different form, i.e. dense, sparse, with 'numpy', 'scipy', or 'torch'"""

    module = sparse

    def __init__(self,*argc,**argv):
        # https://realpython.com/python-super/
        super().__init__(*argc,**argv)

        self.blocks = None
        self.n_blocks = None
        self.eigenvalues = None
        self.eigenstates = None
        self.nearly_diag = None
        self.is_adjacency = None
        # if is_adjacency:
        #     self.is_adjacency = True
        # else:
        #     self.is_adjacency = self.det_is_adjacency()
    
    def clone(self,*argc,**argv)->T:
        return type(self)(*argc,**argv)
        

    def save(self,file):
        if matrix.module is sparse :
            sparse.save_npz(file,self)
        else :
            raise ImplErr
    
    @staticmethod
    def load(file):
        if matrix.module is sparse :
            return sparse.load_npz(file)
        else :
            raise ImplErr

    @classmethod
    def diags(cls,*argc,**argv):
        """diagonal matrix"""
        return cls(matrix.module.diags(*argc,**argv))

    @classmethod
    def kron(cls,*argc,**argv):
        """kronecker product"""
        return cls(matrix.module.kron(*argc,**argv))

    @classmethod
    def identity(cls,*argc,**argv):
        """identity operator"""
        return cls(matrix.module.identity(*argc,**argv))
    
    def dagger(self)->T:
        return self.clone(self.conjugate().transpose()) #type(self)(self.conjugate().transpose())

    def is_symmetric(self,**argv)->bool:
        if matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.transpose()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_hermitean(self,**argv)->bool:
        if matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.dagger()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_unitary(self,**argv)->bool:
        if matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self @ self.dagger() - self.identity(len(self)) ).norm() < tolerance
        else :
            raise ImplErr
        
    def det_is_adjacency(self):
        if self.is_adjacency is None:
            self.is_adjacency = self == self.adjacency()
            
        return self.is_adjacency

    @staticmethod
    def commutator(A,B):
        C = A @ B - B @ A 
        return C
    
    def commute(self,A,tol=1e-6)->bool:
        return matrix.commutator(self,A).norm() < tol

    # @staticmethod 
    def norm(self)->float:
        if matrix.module is sparse :
            return sparse.linalg.norm(self)
        else :
            raise ImplErr

    def adjacency(self)->T:
        if matrix.module is sparse :
            data    = self.data
            indices = self.indices
            indptr  = self.indptr
            data = data.real.astype(int)
            data.fill(1)
            out = self.clone((data, indices, indptr),self.shape)
            if self.blocks is not None:
                out.blocks = self.blocks
            if self.n_blocks is not None:
                out.n_blocks = self.n_blocks
            out.is_adjacency = True
            return out # type(self)(dtype((data, indices, indptr),self.shape))
        else :
            raise ImplErr
    
    def sparsity(self)->float:
        if matrix.module is sparse :
            rows, cols = self.nonzero()
            shape = self.shape
            return float(len(rows)) / float(shape[0]*shape[1])
        else :
            raise ImplErr
        # adjacency = self.adjacency()
        # v = adjacency.flatten()
        # return float(matrix.norm(v)) / float(len(v))

    def visualize(self,adjacency=True,tab='tab10',cb=True,file=None)->None:
        if adjacency:
            return self.adjacency().visualize(False)
        
        import matplotlib.pyplot as plt  
        # from matplotlib.colors import ListedColormap
        # Create a figure and axis
        fig, ax = plt.subplots()  
        M = self.todense()
        C = copy(M).astype(float)
        C.fill(np.nan)
        if self.blocks is None and cb:
            self.count_blocks()
        if self.blocks is not None:
            from matplotlib.colors import LinearSegmentedColormap
            for n in range(len(self)):
                # replace = copy(M[n,:]) * ( self.blocks[n] + 1 )
                C[n,:] = self.blocks[n] + 1 #replace
                C[:,n] = self.blocks[n] + 1 #replace.reshape((-1,1))
            # Create a colormap with N+1 colors, where the first color is white
            N = self.n_blocks+1
            # colors = plt.cm.get_cmap('tab10', N)
            # colors_list = [colors(i) for i in range(N)]
            # custom_cmap = ListedColormap(colors_list)
            # Get the 'tab10' colormap
            tab10_cmap = plt.cm.get_cmap(tab)

            # Define the new colormap
            colors = tab10_cmap(np.linspace(0, 1, N))  # Use 11 colors for tab10
            colors[0] = [1, 1, 1, 1]  # Set the first color (0) to white

            # Create the custom colormap
            custom_cmap = LinearSegmentedColormap.from_list('custom_tab10', colors, N=N)
        else :
            custom_cmap = "binary"
        # colors = copy(M).astype(object).fill("red")
        ax.matshow(np.multiply(M,C), cmap=custom_cmap,origin='upper',extent=[0, M.shape[1], M.shape[0], 0]) # ,facecolors=colors)
        ax.xaxis.set(ticks=np.arange(0.5, M.shape[1]), ticklabels=np.arange(0,M.shape[1], 1))
        ax.yaxis.set(ticks=np.arange(0.5, M.shape[0]), ticklabels=np.arange(0,M.shape[0], 1))
        argv = {
            "linewidth":0.5,
            "linestyle":'--',
            "color":"blue",
            "alpha":0.8
        }
        for x in np.arange(0,M.shape[1]+1, 1):
            ax.axhline(x, **argv) # horizontal lines
        for y in np.arange(0,M.shape[0]+1, 1):
            ax.axvline(y, **argv) # horizontal lines
        plt.ylabel("rows")
        plt.title("columns")
        plt.tight_layout()
        if file is None:
            plt.show()
        else:
            plt.savefig(file)
        return

    def __repr__(self):
        string  = "{:>12s}: {}\n".format('type', str(self.data.dtype))
        string += "{:>12s}: {}\n".format('shape', str(self.shape))
        string += "{:>12s}: {:6f}\n".format('sparsity', self.sparsity())
        string += "{:>12s}: {}\n".format('# all', str(self.count("all")))
        string += "{:>12s}: {}\n".format('#  on', str(self.count("diag")))
        string += "{:>12s}: {}\n".format('# off', str(self.count("off")))
        string += "{:>12s}: {:6f}\n".format('norm', self.norm())
        string += "{:>12s}: {}\n".format('unitary', str(self.is_unitary()))
        string += "{:>12s}: {}\n".format('hermitean', str(self.is_hermitean()))
        string += "{:>12s}: {}\n".format('symmetric', str(self.is_symmetric()))
        return string
    
    def __len__(self)->int:
        M,N = self.shape
        if M != N :
            raise ValueError("matrix is not square: __len__ is not well defined")
        return M
    
    def empty(self)->T:
        return self.clone(self.shape,dtype=self.dtype) # type(self)(self.shape,dtype=self.dtype)
    
    def as_diagonal(self)->T:
        diagonal_elements = super().diagonal()
        return type(self).diags(diagonals=diagonal_elements)
    

    def off_diagonal(self)->T:
        diagonal_elements = super().diagonal() # self.diagonal()
        diagonal_matrix = type(self).diags(diagonals=diagonal_elements)
        return self - diagonal_matrix
    
    def count(self,what="all")->int:
        if what == "all":
            a = self.adjacency()
            return len(a.data)
        elif what == "off":
            return self.off_diagonal().count("all")
        elif what == "diag":
            return self.as_diagonal().count("all")
        else :
            raise ImplErr
        
    @classmethod
    def from_blocks(cls,blocks):
        N = len(blocks)
        tmp = np.full((N,N),None,dtype=object)
        for n in range(N):
            tmp[n,n] = blocks[n]
        if matrix.module is sparse : 
            return cls(sparse.bmat(tmp))
        else :
            raise ValueError("error")

    def count_blocks(self,inplace=True):
        adjacency = self.adjacency()
        if matrix.module is sparse : 
            n_components, labels = connected_components(adjacency,directed=False,return_labels=True)
            if inplace:
                self.blocks = labels
                self.n_blocks = len(np.unique(labels))
            return n_components, labels
        else :
            raise ImplErr

    def mask2submatrix(self,mask)->T:
        submatrix = self[mask][:, mask] #

        # this is needed to restart from a previous calculation
        if self.eigenvalues is not None :
            submatrix.eigenvalues = self.eigenvalues[mask]
        if self.eigenstates is not None :
            submatrix.eigenstates = self.eigenstates[mask][:, mask]
        if self.nearly_diag is not None :
            submatrix.nearly_diag = self.nearly_diag[mask][:, mask]
            
        return submatrix
    
    # @numba.jit
    def diagonalize_each_block(self:T,labels:np.ndarray,method:str,original:bool,tol:float,max_iter:int)->Union[np.ndarray,T]:
        
        # we need to specify all the parameters if we want to speed it up with 'numba.jit'

        # if not original :
        #     raise ValueError("some error occurred")
        
        submatrices = np.full((self.n_blocks,self.n_blocks),None,dtype=object)
        eigenvalues = np.full(self.n_blocks,None,dtype=object)
        eigenstates = np.full(self.n_blocks,None,dtype=object)

        if original :
            indeces = np.arange(self.shape[0])
            permutation = np.arange(self.shape[0])
            k = 0
            print("\tStarting diagonalization")

        for n in numba.prange(self.n_blocks):
            if original : print("\t\tdiagonalizing block n. {:d}".format(n))

            mask = (labels == n)
            permutation[k:k+len(indeces[mask])] = indeces[mask]
            k += len(indeces[mask])
            
            # create a submatrix from one block
            submatrix = self.mask2submatrix(mask)

            # diagonalize the block
            v,f,M = submatrix.eigensolver(  original=False,
                                            method=method,
                                            tol=tol,
                                            max_iter=max_iter)
            submatrices[n,n] = M
            eigenvalues[n] = v
            eigenstates[n] = f
        
        eigenvalues = np.concatenate(eigenvalues)
        eigenstates = matrix.from_blocks(eigenstates)

        reverse_permutation = np.argsort(permutation)
        eigenvalues = eigenvalues[reverse_permutation]
        eigenstates = eigenstates[reverse_permutation][:, reverse_permutation]
        nearly_diagonal = self.clone(bmat(submatrices))[reverse_permutation][:, reverse_permutation]
        # type(self)(bmat(submatrices))
        return eigenvalues, eigenstates, nearly_diagonal
    

    def eigensolver(self,method="jacobi",original=True,tol:float=1.0e-3,max_iter:int=-1):

        # if matrix.module is sparse :

        #############################
        # |-------------------------|
        # |          |   original   |
        # | n_blocks | True | False |
        # |----------|--------------|
        # |    =1    |  ok  |  yes  |
        # |    >1    |  ok  | error |
        # |-------------------------|
        
        N = None

        n_components, labels = self.count_blocks(inplace=True)               
        # self.blocks = labels
        # self.n_blocks = len(np.unique(labels))
        # self.visualize()
        if original : 
            print("\tn_components:",n_components) 
        elif self.n_blocks != 1 :
            raise ValueError("some error occurred")

        if self.n_blocks == 1 :
            if self.shape[0] < NoJacobi:
                method = "dense"
            match method:
                case "jacobi":
                    if not self.is_hermitean():
                        raise ValueError("'matrix' object is not hermitean and then it can not be diagonalized with the Jacobi method")
                    from QuantumSparse.matrix.jacobi import jacobi
                    w,f,N = jacobi(self,tol=tol,max_iter=max_iter)
                case "dense":
                    M = np.asarray(self.todense())
                    if self.is_hermitean():
                        w,f = eigh(M)
                    else :
                        w,f = eig(M)
                    N = self.empty()
                    N.setdiag(w)
                case _:
                    raise ImplErr
        
        elif self.n_blocks > 1:
            w,f,N = self.diagonalize_each_block(labels=labels,original=True,method=method,tol=tol,max_iter=max_iter)

        else :
            raise ValueError("error: n. of block should be >= 1")
        
        self.eigenvalues = copy(w)
        self.eigenstates = copy(f)
        self.nearly_diag = copy(N) if N is not None else None
        
        return w,f,N