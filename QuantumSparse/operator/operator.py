# "operator" class
import numpy as np
import pickle
from QuantumSparse.matrix import matrix

class operator(matrix):
   
    def __init__(self,*argc,**argv):
        super().__init__(*argc,**argv)
        pass

    def save(self,file):
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
    def commutator(A,B):
        C = A @ B - B @ A 
        return C
    
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

    def diagonalize(self,method="jacobi",restart=False,tol:float=1.0e-3,max_iter:int=-1):

        if restart :
            self.eigenvalues = None
            self.eigenstates = None
            self.nearly_diag = None

        if not self.is_hermitean():
            raise ValueError("'operator' is not hermitean")
        
        w,f,_ = super().diagonalize(method=method,original=True,tol=tol,max_iter=max_iter)
        return w,f
        # self.eigenvalues = w
        # self.eigenstates = f
        # self.nearly_diag = N

        # return self.eigenvalues, self.eigenstates
    
