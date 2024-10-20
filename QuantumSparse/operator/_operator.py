# "operator" class

import numpy as np
import pickle
from copy import copy
from QuantumSparse.matrix import Matrix
from typing import TypeVar, Union, List

T = TypeVar('T', bound='Operator')  # type of the class itself


class Operator(Matrix):
    """
    This class is a subclass of QuantumSparse.matrix.Matrix and is used to represent a general operator.
    """
    
    name:str
    
    
    def __mul__(self: T, other: T) -> T:
        """
        Implements the left matrix multiplication operator (*) for the Operator class.

        Parameters
        ----------
        self : Operator
            The Operator object to multiply.
        other : Operator
            The Operator object to multiply with.

        Returns
        -------
        T
            The result of the matrix multiplication.
        """
        return super().__mul__(other)
    
    def __rmul__(self: T, other: T) -> T:
        """
        Implements the right matrix multiplication operator (*) for the Operator class.

        Parameters
        ----------
        self : Operator
            The Operator object to multiply.
        other : Operator
            The Operator object to multiply with.

        Returns
        -------
        T
            The result of the matrix multiplication.
        """
        return super().__rmul__(other)
    
    def __matmul__(self,other):
        """
        Implements the matrix multiplication operator (@) for the Operator class.

        Parameters
        ----------
        other : Operator
            The Operator object to multiply with.

        Returns
        -------
        Operator
            The result of the matrix multiplication.
        """
        return super().__matmul__(other)

    def __init__(self: T, *argc, **argv) -> None:
        """
        Initialize the Operator object.

        Parameters
        ----------
        argc : tuple
            Positional arguments passed to the constructor.
        argv : dict
            Keyword arguments passed to the constructor.

        Returns
        -------
        None
        """
        super().__init__(*argc, **argv)
        self.name = None
        pass

    def __repr__(self: T) -> str:
        string = "{:>12s}: {}\n".format('name', str(self.name))
        string += super().__repr__()
        return string
        
    # def save(self: T, file: str) -> None:
    #     """
    #     Save the Operator object to a file.

    #     Parameters
    #     ----------
    #     file : str
    #         The file path to save the object.

    #     Returns
    #     -------
    #     None
    #     """
    #     with open(file, 'wb') as f:
    #         pickle.dump(self, f)

    # @classmethod
    # def load(cls: Type[T], file: str) -> T:
    #     """
    #     Load an Operator object from a file.

    #     Parameters
    #     ----------
    #     file : str
    #         The file path to load the object.

    #     Returns
    #     -------
    #     T
    #         The loaded Operator object.
    #     """
    #     with open(file, 'rb') as f:
    #         obj = pickle.load(f)
    #     return cls(obj)

    
    @staticmethod
    def identity(dimensions)->T:
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
            iden = Operator.identity([dimensions])[0]
            return iden
        else :            
            N = len(dimensions)
            iden = np.zeros(N,dtype=object)
            for i,dim in zip(range(N),dimensions):
                iden[i] = Matrix.identity(dim,dtype=int)  
            return iden
    
    def diagonalize(self:T,restart=False,tol:float=1.0e-3,max_iter:int=-1,test=True):
        """
        Diagonalize the operator using the specified method.

        Parameters
        ----------
        restart : bool, optional
            Whether to restart the diagonalization process (default is False).
        tol : float, optional
            The tolerance for the diagonalization process (default is 1.0e-3).
        max_iter : int, optional
            The maximum number of iterations for the diagonalization process (default is -1).
        test : bool, optional
            Whether to test the eigensolution (default is True).

        Returns
        -------
        w : numpy.ndarray
            The eigenvalues of the operator.
        f : numpy.ndarray
            The eigenstates of the operator.
        """

        if restart :
            self.eigenvalues = None
            self.eigenstates = None
            ##NEARLY_DIAG## self.nearly_diag = None

        # if not self.is_hermitean():
        #     raise ValueError("'operator' is not hermitean")
        
        ##NEARLY_DIAG##w,f,_ = super().eigensolver(method=method,original=True,tol=tol,max_iter=max_iter)
        self.eigenvalues,self.eigenstates = super().eigensolver(original=True,tol=tol,max_iter=max_iter)
        if test:
            eigtest = self.test_eigensolution()
            print("\teigensolution test:",eigtest.norm())
        # self.eigenvalues = np.asarray(self.eigenvalues)
        # self.eigenstates = np.asarray(self.eigenstates)
        
        self.eigenstates.n_blocks = self.n_blocks
        self.eigenstates.blocks = self.blocks
        
        # print(f)
        return self.eigenvalues,self.eigenstates
        # self.eigenvalues = w
        # self.eigenstates = f
        # self.nearly_diag = N

        # return self.eigenvalues, self.eigenstates
        
        

    def change_basis(self:T,S:T,direction="forward"):
        """
        Changes the basis of the operator using the given symmetry operator.

        Parameters
        ----------
        S : T
            The symmetry operator used for the basis change.
        direction : str, optional
            The direction of the basis change (default is "forward").

        Returns
        -------
        out : T
            The operator in the new basis.
        """

        if not S.diagonalized():
            raise ValueError("The operator 'S' should have already been diagonalized.")
        
        if direction == "forward":
            out = self.clone(S.eigenstates.dagger() @ self @ S.eigenstates)
            if self.diagonalized():
                out.eigenvalues = copy(self.eigenvalues)
                out.eigenstates = S.eigenstates.dagger() @ self.eigenstates
                ##NEARLY_DIAG## out.nearly_diag = S.eigenstates.dagger() @ self.nearly_diag @ S.eigenstates
        
        elif direction == "backward":
            out = self.clone(S.eigenstates @ self @ S.eigenstates.dagger())
            if self.diagonalized():
                out.eigenvalues = copy(self.eigenvalues)
                out.eigenstates = S.eigenstates @ self.eigenstates
                ##NEARLY_DIAG## out.nearly_diag = S.eigenstates @ self.nearly_diag @ S.eigenstates.dagger() 

        else:
            raise ValueError("'direction' can be only 'forward' or 'backward'")
        
        return out


    def diagonalize_with_symmetry(self:T,S:Union[List[T],T],use_block_form=True,test=True,**argv):
        """
        Diagonalizes the operator using the given symmetry operator(s).

        Parameters
        ----------
        S : Union[List[T], T]
            The symmetry operator(s) used for the diagonalization.
            If a single operator is provided, it will be wrapped in a list.
        use_block_form : bool, optional
            Whether to use the block form of the operator (default is False).
        test : bool, optional
            Whether to test the eigensolution (default is True).
        **argv
            Additional keyword arguments to be passed to the diagonalization method.

        Returns
        -------
        Tuple[T, T]
            A tuple containing the eigenvalues and eigenstates of the operator.
        """

        if type(S) is not list:
            return self.diagonalize_with_symmetry([S],use_block_form,test,**argv)
        if len(S) == 0 :
            return self.diagonalize(**argv)
        
        sym = S[0]

        if not self.commute(sym):
            raise ValueError('Ypu provided a symmetry operator which does not commute with the operator that you want to diagonalize.')
        
        if sym.eigenvalues is None:
            raise ValueError("The symmetry operator 'S' should have already been diagonalized.")

        # I should define a 'symmetry' operator
        w,labels = unique_with_tolerance(sym.eigenvalues)
        
        # new = self.clone(sym.eigenstates.dagger() @ self @ sym.eigenstates)
        to_diag:T = self.change_basis(sym,direction="forward")
        for n in range(1,len(S)):
            S[n] = S[n].change_basis(sym,direction="forward")

        # little dirty trick
        def new_count_blocks(self:T,inplace=True)->T:
            self.blocks = labels
            self.n_blocks = len(np.unique(labels))
            return self.n_blocks, self.blocks
        import types
        to_diag.count_blocks  = types.MethodType(new_count_blocks, to_diag)

        if use_block_form:
            to_diag.count_blocks(inplace=True)
            to_diag = to_diag.divide_into_block(labels)
        # else:
        #     to_diag = new

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
        ##NEARLY_DIAG## self.nearly_diag = to_diag.nearly_diag # @ to_diag.nearly_diag @ S.eigenstates.dagger()

        if test:
            solution = self.test_eigensolution()
            norm = solution.norm()
            print("\teigensolution test:",norm)
        
        return self.eigenvalues, self.eigenstates
        # return copy(self.eigenvalues),copy(self.eigenstates)
    
    def energy_levels(self,tol=1e-8):
        
        if self.eigenvalues is None:
            raise ValueError("The operator has not been diagonalized yet.")
        
        w,index = unique_with_tolerance(self.eigenvalues,tol)

        return w,np.asarray([ (index==a).sum() for a in range(len(w)) ])
    
    def band_diagram(self:T,sym:T):
        
        if not self.diagonalized():
            raise ValueError("The operator has not been diagonalized yet.")
        
        if not self.commute(sym):
            raise ValueError('Ypu provided a symmetry operator which does not commute with the operator that you want to diagonalize.')
        
        if sym.eigenvalues is None:
            raise ValueError("The symmetry operator 'S' should have already been diagonalized.")
        
        new:T = self.change_basis(sym,direction="forward")

        w,labels = unique_with_tolerance(sym.eigenvalues)
        # little dirty trick
        def new_count_blocks(self:T,inplace=True)->T:
            self.blocks = labels
            self.n_blocks = len(np.unique(labels))
            return self.n_blocks, self.blocks
        import types
        new.count_blocks  = types.MethodType(new_count_blocks, new)
        new.count_blocks(inplace=True)
        
        # divide_into_block
        
        # submatrices = np.full((self.n_blocks, self.n_blocks), None, dtype=object)
        submatrices = [None]*new.n_blocks
        indeces = np.arange(new.shape[0])
        permutation = np.arange(new.shape[0])

        k = 0
        for n in range(new.n_blocks):
            mask = (labels == n)
            permutation[k:k + len(indeces[mask])] = indeces[mask]
            k += len(indeces[mask])
            # create a submatrix from one block
            # submatrices[n,n] = self.mask2submatrix(mask)
            submatrices[n] = new.mask2submatrix(mask)
        
        assert len(submatrices) == len(w), "coding error"

        return w, submatrices
        



def unique_with_tolerance(arr, tol=1e-8):
    """
    Returns the unique elements of an array within a specified tolerance.

    Parameters:
    arr (array_like): The input array.
    tol (float, optional): The tolerance for uniqueness. Defaults to 1e-8.

    Returns:
    tuple: A tuple containing the unique elements and their indices.
    """
    rounded_arr = np.round(arr, decimals=int(-np.log10(tol)))
    unique_rounded,index = np.unique(rounded_arr,return_inverse=True)
    return unique_rounded, index
