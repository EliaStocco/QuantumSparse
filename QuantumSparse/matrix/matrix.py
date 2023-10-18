import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
# from . import dtype
from ..tools.functional import output
import numpy as np
from copy import copy
import numba 

dtype = csr_matrix

class get_class(type):
    def __new__(cls, name, bases, attrs):
        # Iterate through the attributes of the class
        if issubclass(dtype, sparse.spmatrix) :
            attrs["module"] = sparse
        else :
            raise ValueError("not implemented yet")
        
        bases = bases + (dtype,)

        # Create the class using the modified attributes
        return super().__new__(cls, name, bases, attrs)
    

class matrix(metaclass=get_class):
# class matrix(csr_matrix):
    """class to handle matrices in different form, i.e. dense, sparse, with 'numpy', 'scipy', or 'torch'"""

    module = sparse

    def __init__(self,*argc,**argv):
        # https://realpython.com/python-super/
        super().__init__(*argc,**argv)

        self.blocks = None
        pass

    #@staticmethod
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
    
    def dagger(self):
        return self.conjugate().transpose()

    def is_hermitean(self,**argv):

        if matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return scipy.sparse.linalg.norm(self - self.dagger()) < tolerance
        else :
            raise ValueError("not implemented yet")

    @staticmethod 
    def norm(obj):
        if matrix.module is sparse :
            return sparse.linalg.norm(obj)
        else :
            raise ValueError("not implemented yet")
        
        
    def adjacency(self):
        if matrix.module is sparse :
            tmp = dtype(self.shape,dtype=int)
            rows, cols = self.nonzero()
            for r,c in zip(rows, cols):
                tmp[r,c] = 1
            return tmp
            # # Assuming you have a sparse matrix called 'sparse_matrix' in CSR format
            # # Create a new CSR matrix with 1s where the original matrix is non-zero
            # return dtype(   (self.data != 0).astype(int),\
            #                 indices=self.indices,\
            #                 indptr=self.indptr,\
            #                  shape=self.shape)
        else :
            raise ValueError("not implemented yet")
    
    def sparsity(self):
        if matrix.module is sparse :
            rows, cols = self.nonzero()
            shape = self.shape
            return float(len(rows)) / float(shape[0]*shape[1])
        # adjacency = self.adjacency()
        # v = adjacency.flatten()
        # return float(matrix.norm(v)) / float(len(v))

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

        
    def diagonalize(self,original=True):

        adjacency = self.adjacency()
        # eigenvalues = np.array(self.shape[0])
        # eigenfunctions = 

        if matrix.module is sparse :
            n_components, labels = sparse.csgraph.connected_components(adjacency,directed=False,return_labels=True)
            # del adjacency
            if original : print("\tn_components:",n_components)
            if original : print("\tlabels:",labels)
            # print(adjacency.todense())

            self.blocks = labels
            self.n_blocks = len(np.unique(labels))

            if self.n_blocks == 1 :
                M = np.asarray(self.todense())
                v,f = np.linalg.eigh(M)
                return v,f
            elif self.n_blocks > 1:
                # this should be parallelized by numba
                submatrices = np.full((self.n_blocks,self.n_blocks),None,dtype=object)
                eigenvalues = np.full(self.n_blocks,None,dtype=object)
                eigenstates = np.full(self.n_blocks,None,dtype=object)
                if original :
                    indeces = np.arange(self.shape[0])
                    permutation = np.arange(self.shape[0])
                    k = 0
                    print("\tStarting diagonalization")
                for n in numba.prange(self.n_blocks):
                    if original : print("\tdiagonalizing block ",n)
                    mask = (labels == n)
                    submatrix = matrix(self[mask][:, mask])
                    submatrices[n,n] = submatrix
                    v,f = submatrix.diagonalize(original=False)
                    eigenvalues[n] = v
                    eigenstates[n] = f
                    if original :
                        permutation[k:k+len(indeces[mask])] = indeces[mask]
                        k += len(indeces[mask])
                
                # reordered_matrix = self[permutation][:, permutation]
                # block = matrix(sparse.bmat(submatrices))
                # np.linalg.norm((block - reordered_matrix).todense()) # = 0.0
                if original:
                    reverse_permutation = np.argsort(permutation)

                # M = matrix(block).adjacency()
                # a,b = sparse.csgraph.connected_components(M,directed=False,return_labels=True)
                # print(np.all(b[:-1] <= b[1:])) # this check whether 'b' is sorted. It should be True

                eigenvalues = matrix(np.concatenate(eigenvalues)) #matrix.from_blocks(eigenvalues)
                eigenstates = matrix.from_blocks(eigenstates)

                if original :
                    print("testing")

                    # w = eigenvalues[:,reverse_permutation]
                    # f = eigenstates[reverse_permutation][:, reverse_permutation]

                    # delta = sparse.linalg.norm(self @ f - f.multiply(w))

                    return eigenvalues, eigenstates, reverse_permutation
                else :
                    return eigenvalues, eigenstates

            else :
                raise ValueError("error")
                

            print("testing")

        else :
            raise ValueError("not implemented yet")

    