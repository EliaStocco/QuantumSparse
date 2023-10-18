from scipy.linalg import eigh
from scipy import sparse
from scipy.sparse.linalg import eigsh
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

        pass

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
    
    def dagger(self:T)->T:
        return type(self)(self.conjugate().transpose())

    def is_symmetric(self:T,**argv)->bool:
        if matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.transpose()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_hermitean(self:T,**argv)->bool:
        if matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.dagger()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_unitary(self:T,**argv)->bool:
        if matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self @ self.dagger() - self.identity(len(self)) ).norm() < tolerance
        else :
            raise ImplErr

    # @staticmethod 
    def norm(self):
        if matrix.module is sparse :
            return sparse.linalg.norm(self)
        else :
            raise ImplErr

    def adjacency(self):
        if matrix.module is sparse :
            data    = self.data
            indices = self.indices
            indptr  = self.indptr
            data = data.real.astype(int)
            data.fill(1)
            return type(self)(dtype((data, indices, indptr),self.shape))
        else :
            raise ImplErr
    
    def sparsity(self):
        if matrix.module is sparse :
            rows, cols = self.nonzero()
            shape = self.shape
            return float(len(rows)) / float(shape[0]*shape[1])
        else :
            raise ImplErr
        # adjacency = self.adjacency()
        # v = adjacency.flatten()
        # return float(matrix.norm(v)) / float(len(v))

    def __len__(self)->int:
        M,N = self.shape
        if M != N :
            raise ValueError("matrix is not square: __len__ is not well defined")
        return M
    
    def empty(self:T)->T:
        return type(self)(self.shape,dtype=self.dtype)
    
    def as_diagonal(self:T)->T:
        diagonal_elements = super().diagonal()
        return type(self).diags(diagonals=diagonal_elements)
    

    def off_diagonal(self:T)->T:
        diagonal_elements = super().diagonal() # self.diagonal()
        diagonal_matrix = type(self).diags(diagonals=diagonal_elements)
        return self - diagonal_matrix
    
    def count(self:T,what="all")->int:
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

    def count_blocks(self,return_labels=True):
        adjacency = self.adjacency()
        if matrix.module is sparse : 
            return connected_components(adjacency,directed=False,return_labels=return_labels)
        else :
            raise ImplErr
        
    def mask2submatrix(self,mask):
        submatrix = matrix(self[mask][:, mask])

        # this is needed to restart from a previous calculation
        if self.eigenvalues is not None :
            submatrix.eigenvalues = self.eigenvalues[mask]
        if self.eigenstates is not None :
            submatrix.eigenstates = self.eigenstates[mask][:, mask]
        if self.nearly_diag is not None :
            submatrix.nearly_diag = self.nearly_diag[mask][:, mask]
            
        return submatrix

    @numba.jit
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
            v,f,M = submatrix.diagonalize(original=False,
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
        nearly_diagonal = type(self)(bmat(submatrices))[reverse_permutation][:, reverse_permutation]
        return eigenvalues, eigenstates, nearly_diagonal
    

    def diagonalize(self,method="jacobi",original=True,tol:float=1.0e-3,max_iter:int=-1):

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

        n_components, labels = self.count_blocks(return_labels=True)               
        self.blocks = labels
        self.n_blocks = len(np.unique(labels))
        if original : 
            print("\tn_components:",n_components) 
        elif self.n_blocks != 1 :
            raise ValueError("some error occurred")

        if self.n_blocks == 1 :
            match method:
                case "jacobi":
                    from QuantumSparse.matrix.jacobi import jacobi
                    w,f,N = jacobi(self,tol=tol,max_iter=max_iter)
                case "dense":
                    M = np.asarray(self.todense())
                    w,f = eigh(M)
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
