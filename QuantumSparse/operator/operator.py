# "operator" class
import numpy as np
import pickle
from copy import copy
from QuantumSparse.matrix import matrix
from typing import TypeVar
T = TypeVar('T') 

class operator(matrix):
   
    def __init__(self,*argc,**argv):
        super().__init__(*argc,**argv)
        pass

    def save(self:T,file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def load(cls,file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        return obj

    
    @staticmethod
    def identity(dimensions):
        """
        Parameters
        ----------
        dimensions : numpy.array
            numpy.array of integer numbers representing the Hilbert space dimension of each site 

        Returns
        -------
        iden : numpy.array of scipy.sparse
            array of the identity operator for each site, represented with sparse matrices,
            acting on the local (only one site) Hilbert space
        """
        if not hasattr(dimensions, '__len__'):
            iden = operator.identity([dimensions])[0]
            return iden
        else :            
            N = len(dimensions)
            iden = np.zeros(N,dtype=object)
            for i,dim in zip(range(N),dimensions):
                iden[i] = matrix.identity(dim,dtype=int)  
            return iden
    
    @staticmethod
    def anticommutator(A,B):
        C = A @ B + B @ A 
        return C
    
    def eigen(self):
        return {"eigenvalues":self.eigenvalues,"eigenstates":self.eigenstates}
    
    def test_diagonalization(self,tol=1e-6,return_norm=False):
        """Test the accuracy of the eigen-ecomposition"""
        test = self @ self.eigenstates - self.eigenstates @ self.diags(diagonals=self.eigenvalues,shape=self.shape)
        norm = test.norm()
        if return_norm:
            return norm < tol, norm
        else :
            return norm < tol

    def diagonalize(self,method="jacobi",restart=False,tol:float=1.0e-3,max_iter:int=-1,test=True):

        if restart :
            self.eigenvalues = None
            self.eigenstates = None
            self.nearly_diag = None

        # if not self.is_hermitean():
        #     raise ValueError("'operator' is not hermitean")
        
        w,f,_ = super().eigensolver(method=method,original=True,tol=tol,max_iter=max_iter)
        if test:
            print("\teigensolution test:",self.test_eigensolution().norm())
        return w,f
        # self.eigenvalues = w
        # self.eigenstates = f
        # self.nearly_diag = N

        # return self.eigenvalues, self.eigenstates

    def diagonalize_with_symmetry(self,S,use_block_form=False,test=True,**argv):

        w,labels = unique_with_tolerance(S.eigenvalues)
        # ( abs(w[ii] - S.eigenvalues) > 1e-8 ).sum()

        # np.linalg.norm((S.eigenstates.dagger() @ S @ S.eigenstates ).diagonal() - S.eigenvalues ) = 1e-14
        # (S.eigenstates.dagger() @ S @ S.eigenstates ).off_diagonal().norm() = 1e-14
        
        new = self.clone(S.eigenstates.dagger() @ self @ S.eigenstates)
        # new.is_hermitean() = True

        # little dirty trick
        # a,b = new.count_blocks()
        def new_count_blocks(self,inplace=True):
            self.blocks = labels
            self.n_blocks = len(np.unique(labels))
            return self.n_blocks, self.blocks
        import types
        new.count_blocks  = types.MethodType(new_count_blocks, new)

        # new.count_blocks()
        if use_block_form:
            to_diag = new.divide_into_block(labels)
        else:
            to_diag = new
        # test = (new - newB).norm()

        # import matplotlib.pyplot as plt
        # plt.imshow(np.absolute(S.todense())>0.5,cmap="tab10")
        # plt.show()

        w,f = to_diag.diagonalize(restart=True,test=False,**argv)

        
        # self = self.clone(S.eigenstates @ new @ S.eigenstates.dagger())

        self.eigenvalues = to_diag.eigenvalues
        self.eigenstates = S.eigenstates @ to_diag.eigenstates # @ S.eigenstates.dagger()
        self.nearly_diag = S.eigenstates @ to_diag.nearly_diag @ S.eigenstates.dagger()

        if test:
            print("\teigensolution test:",self.test_eigensolution().norm())
        
        return copy(self.eigenvalues),copy(self.eigenstates)



def unique_with_tolerance(arr, tol=1e-8):
    rounded_arr = np.round(arr, decimals=int(-np.log10(tol)))
    unique_rounded,index = np.unique(rounded_arr,return_inverse=True)
    return unique_rounded, index
