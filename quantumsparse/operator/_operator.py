import numpy as np
from copy import copy, deepcopy
from typing import TypeVar, Union, List
from quantumsparse.matrix import Matrix
from quantumsparse.tools.mathematics import unique_with_tolerance
from quantumsparse.bookkeeping import TOLERANCE

T = TypeVar('T', bound='Operator')  # type of the class itself

OpArr = Union[List[T],T]

class Operator(Matrix):
    """
    This class is a subclass of quantumsparse.matrix.Matrix and is used to represent a general operator.
    """
    
    name:str
    eigenstates:T
    
    
    # def __mul__(self: T, other: T) -> T:
    #     """
    #     Implements the left matrix multiplication operator (*) for the Operator class.

    #     Parameters
    #     ----------
    #     self : Operator
    #         The Operator object to multiply.
    #     other : Operator
    #         The Operator object to multiply with.

    #     Returns
    #     -------
    #     T
    #         The result of the matrix multiplication.
    #     """
    #     return super().__mul__(other)
    
    # def __rmul__(self: T, other: T) -> T:
    #     """
    #     Implements the right matrix multiplication operator (*) for the Operator class.

    #     Parameters
    #     ----------
    #     self : Operator
    #         The Operator object to multiply.
    #     other : Operator
    #         The Operator object to multiply with.

    #     Returns
    #     -------
    #     T
    #         The result of the matrix multiplication.
    #     """
    #     return super().__rmul__(other)
    
    # def __matmul__(self,other):
    #     """
    #     Implements the matrix multiplication operator (@) for the Operator class.

    #     Parameters
    #     ----------
    #     other : Operator
    #         The Operator object to multiply with.

    #     Returns
    #     -------
    #     Operator
    #         The result of the matrix multiplication.
    #     """
    #     return super().__matmul__(other)

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
    def identity(dimensions)->Union[T,List[T]]:
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
            iden:List[T] = np.zeros(N,dtype=object)
            for i,dim in zip(range(N),dimensions):
                iden[i] = Matrix.identity(dim,dtype=int)  
            return iden
    
    # def diagonalize(self:T,restart=False,tol:float=1.0e-3,max_iter:int=-1,**argv):
    #     """
    #     Diagonalize the operator using the specified method.

    #     Parameters
    #     ----------
    #     restart : bool, optional
    #         Whether to restart the diagonalization process (default is False).
    #     tol : float, optional
    #         The tolerance for the diagonalization process (default is 1.0e-3).
    #     max_iter : int, optional
    #         The maximum number of iterations for the diagonalization process (default is -1).
    #     test : bool, optional
    #         Whether to test the eigensolution (default is True).

    #     Returns
    #     -------
    #     w : numpy.ndarray
    #         The eigenvalues of the operator.
    #     f : numpy.ndarray
    #         The eigenstates of the operator.
    #     """

    #     if restart :
    #         self.eigenvalues = None
    #         self.eigenstates = None
    #         ##NEARLY_DIAG## self.nearly_diag = None
        
    #     ##NEARLY_DIAG##w,f,_ = super().eigensolver(method=method,original=True,tol=tol,max_iter=max_iter)
    #     self.eigenvalues,self.eigenstates = super().eigensolver(original=True,tol=tol,max_iter=max_iter,**argv)
        
    #     self.eigenstates.n_blocks = self.n_blocks
    #     self.eigenstates.blocks = self.blocks
        
    #     return self.eigenvalues,self.eigenstates

    def change_basis(self:T,S:T,direction="forward")->T:
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
        
        # ToDo: specialize this function for the case when self == S

        if not S.is_diagonalized():
            raise ValueError("The operator 'S' should have already been diagonalized.")
        
        if direction == "forward":
            dagger = S.eigenstates.dagger()
            out = self.clone(dagger @ self @ S.eigenstates)
            if self.is_diagonalized():
                out.eigenvalues = copy(self.eigenvalues)
                out.eigenstates = dagger @ self.eigenstates
                ##NEARLY_DIAG## out.nearly_diag = S.eigenstates.dagger() @ self.nearly_diag @ S.eigenstates
        
        elif direction == "backward":
            out = self.clone(S.eigenstates @ self @ S.eigenstates.dagger())
            if self.is_diagonalized():
                out.eigenvalues = copy(self.eigenvalues)
                out.eigenstates = S.eigenstates @ self.eigenstates
                ##NEARLY_DIAG## out.nearly_diag = S.eigenstates @ self.nearly_diag @ S.eigenstates.dagger() 

        else:
            raise ValueError("'direction' can be only 'forward' or 'backward'")
        
        try:
            out.extras = deepcopy(self.extras)
        except:
            pass
        return out


    def diagonalize_with_symmetry(self:T,S:Union[List[T],T],test=False,**argv):
        """
        Diagonalizes the operator using the given symmetry operator(s).

        Parameters
        ----------
        S : Union[List[T], T]
            The symmetry operator(s) used for the diagonalization.
            If a single operator is provided, it will be wrapped in a list.
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
            return self.diagonalize_with_symmetry([S],test,**argv)
        if len(S) == 0 :
            return self.diagonalize(**argv)
        
        sym = S[0]

        if not self.commute(sym):
            raise ValueError('You have provided a symmetry operator which does not commute with the operator that you want to diagonalize.')
        
        if sym.eigenvalues is None:
            raise ValueError("The symmetry operator 'S' should have already been diagonalized.")

        w,labels = unique_with_tolerance(sym.eigenvalues)
        
        to_diag:T = self.change_basis(sym,direction="forward")
        for n in range(1,len(S)):
            S[n] = S[n].change_basis(sym,direction="forward")

        #  `to_diag` will have the same block form of `sym`
        # and it will be usefull in `to_diag.clean_block_form` to remove 
        # sporious off diagonal elements (numerical noise)
        to_diag.blocks = labels
        to_diag.n_blocks = len(np.unique(labels))
    
        
        # This line removes all the off-diagonal small elements that could create numerical instabilities
        to_diag = to_diag.clean_block_form(labels)
        
        # Pay attention that `to_diag.blocks` and `to_diag.n_blocks` are both `None` now.
        # But this is okay because `to_diag.count_blocks` will be called withing `to_diag.diagonalize` 
        # and more blocks might be found.

        if len(S) == 1 :
            argv["tqdm"] = True
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
        self.extras      = to_diag.extras
        
        ##NEARLY_DIAG## self.nearly_diag = to_diag.nearly_diag # @ to_diag.nearly_diag @ S.eigenstates.dagger()

        if test:
            solution = self.test_eigensolution()
            norm = solution.norm()
            print("\teigensolution test:",norm)
        
        return self.eigenvalues, self.eigenstates
        # return copy(self.eigenvalues),copy(self.eigenstates)
    
    def energy_levels(self,tol=1e-8,return_indices=False):
        
        if self.eigenvalues is None:
            raise ValueError("The operator has not been diagonalized yet.")
        
        w,index = unique_with_tolerance(self.eigenvalues,tol)

        if return_indices:
            return w,np.asarray([ (index==a).sum() for a in range(len(w)) ]), index
        else:
            return w,np.asarray([ (index==a).sum() for a in range(len(w)) ])
    
    def band_diagram(self:T,sym:T):
        
        if not self.is_diagonalized():
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
        
        # clean_block_form
        
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
    
    def thermal_average(self:T,operator:T,temperatures:np.ndarray)->np.ndarray:
        """
        Calculate the thermal average of an operator at a given temperature.

        Parameters
        ----------
        operator : T
            The operator for which to calculate the thermal average.
        temperature : float
            The temperature at which to calculate the thermal average.

        Returns
        -------
        T
            The thermal average of the operator.
        """
        from quantumsparse.statistics.statistical_physics import quantum_thermal_average_value
        if not self.is_diagonalized():
            raise ValueError("The operator has not been diagonalized yet.")        
        return quantum_thermal_average_value(temperatures,self.eigenvalues,operator,self.eigenstates)
        
def test_operator_save_load(tmp_path):
    """
    Test that saving and loading a Matrix preserves its data.
    """
    # Create example data and Matrix instance
    data = np.array([[1, 2], [3, 4]], dtype=float)
    matrix = Operator(data)

    # Define a temporary filename
    file_path = tmp_path / "test_operator.pickle"
    
    matrix.save(str(file_path))
    loaded_matrix = Operator.load(str(file_path))
    test = (matrix-loaded_matrix).norm()
    assert test < TOLERANCE, "error load/save (1)"
    assert type(matrix) == type(loaded_matrix), "error"
    
    matrix.diagonalize()    
    matrix.save(str(file_path))
    loaded_matrix = Operator.load(str(file_path))
    test = (matrix-loaded_matrix).norm()
    assert test < TOLERANCE, "error load/save (2)"
    assert loaded_matrix.is_diagonalized(), "error load/save (3)"
    test1 = matrix.test_eigensolution()
    test2 = loaded_matrix.test_eigensolution()
    test = (test1-test2).norm()
    assert test < TOLERANCE, "error load/save (4)"
    

if __name__ == "__main__":
    test_operator_save_load()