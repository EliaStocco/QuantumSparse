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
    
    
    # def eigen(self):
    #     return {"eigenvalues":self.eigenvalues,"eigenstates":self.eigenstates}
    
    # def test_diagonalization(self,tol=1e-6,return_norm=False):
    #     """Test the accuracy of the eigen-ecomposition"""
    #     test = self @ self.eigenstates - self.eigenstates @ self.diags(diagonals=self.eigenvalues,shape=self.shape)
    #     norm = test.norm()
    #     if return_norm:
    #         return norm < tol, norm
    #     else :
    #         return norm < tol

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

    def change_basis(self,S,direction="forward"):

        if not S.diagonalized():
            raise ValueError("The operator 'S' should have already been diagonalized.")
        
        if direction == "forward":
            out = self.clone(S.eigenstates.dagger() @ self @ S.eigenstates)
            if self.diagonalized():
                out.eigenvalues = copy(self.eigenvalues)
                out.eigenstates = S.eigenstates.dagger() @ self.eigenstates
                out.nearly_diag = S.eigenstates.dagger() @ self.nearly_diag @ S.eigenstates
        
        elif direction == "backward":
            out = self.clone(S.eigenstates @ self @ S.eigenstates.dagger())
            if self.diagonalized():
                out.eigenvalues = copy(self.eigenvalues)
                out.eigenstates = S.eigenstates @ self.eigenstates
                out.nearly_diag = S.eigenstates @ self.nearly_diag @ S.eigenstates.dagger() 

        else:
            raise ValueError("'direction' can be only 'forward' or 'backward'")
        
        return out


    def diagonalize_with_symmetry(self,S,use_block_form=False,test=True,**argv):

        if type(S) is not list:
            return self.diagonalize_with_symmetry([S],use_block_form,test,**argv)
        if len(S) == 0 :
            return self.diagonalize(**argv)
        
        sym = S[0]

        if not self.commute(sym):
            raise ValueError('Ypu provided a symmetry operator which does not commute with the operator that you want to diagonalize.')

        # I should define a 'symmetry' operator
        w,labels = unique_with_tolerance(sym.eigenvalues)
        
        # new = self.clone(sym.eigenstates.dagger() @ self @ sym.eigenstates)
        new = self.change_basis(sym,direction="forward")
        for n in range(1,len(S)):
            S[n] = S[n].change_basis(sym,direction="forward")

        # little dirty trick
        def new_count_blocks(self,inplace=True):
            self.blocks = labels
            self.n_blocks = len(np.unique(labels))
            return self.n_blocks, self.blocks
        import types
        new.count_blocks  = types.MethodType(new_count_blocks, new)

        if use_block_form:
            to_diag = new.divide_into_block(labels)
        else:
            to_diag = new

        if len(S) == 1 :
            to_diag.diagonalize(test=False,**argv)
        else:
            raise ValueError("not implemented yet")
            to_diag.diagonalize(test=False,**argv)
            # to_diag.change_basis(sym,direction="backward")
            for n in range(1,len(S)):
                S[n] = S[n].change_basis(to_diag,direction="forward")

            # raise ValueError("not implemented yet")
            # I should diagonalize the matrix at first, and then call diagonalize_with_symmetry
            # to_diag.diagonalize_with_symmetry(S[1:],use_block_form,test,**argv)

        to_diag = to_diag.change_basis(sym,direction="backward")

        self.eigenvalues = to_diag.eigenvalues
        self.eigenstates = to_diag.eigenstates # @ to_diag.eigenstates
        self.nearly_diag = to_diag.nearly_diag # @ to_diag.nearly_diag @ S.eigenstates.dagger()

        if test:
            print("\teigensolution test:",self.test_eigensolution().norm())
        
        return copy(self.eigenvalues),copy(self.eigenstates)



def unique_with_tolerance(arr, tol=1e-8):
    rounded_arr = np.round(arr, decimals=int(-np.log10(tol)))
    unique_rounded,index = np.unique(rounded_arr,return_inverse=True)
    return unique_rounded, index
