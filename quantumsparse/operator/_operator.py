import numpy as np
from copy import copy, deepcopy
from typing import TypeVar, Union, List, Type
from typing_extensions import Self
from quantumsparse.matrix import Matrix
from quantumsparse.tools.mathematics import unique_with_tolerance

T = TypeVar('T', bound='Operator')

OpArr = Union[List[T],T]

class Operator(Matrix):
    """
    This class is a subclass of quantumsparse.matrix.Matrix and is used to represent a general operator.
    """
    
    @staticmethod
    def identity(dimensions)->OpArr:
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

        if not S.is_diagonalized():
            raise ValueError("The operator 'S' should have already been diagonalized.")
        
        if direction not in ["forward","backward"]:
            raise ValueError("'direction' can be only 'forward' or 'backward'")
        
        out = self.unitary_transformation(S.eigenstates,dagger=direction=="forward")
        
        # if direction == "forward":
        #     dagger = S.eigenstates.dagger()
        #     out = Operator(dagger @ self @ S.eigenstates)
        #     if self.is_diagonalized():
        #         out.eigenstates = dagger @ self.eigenstates
        
        # elif direction == "backward":
            
        #     # out = Operator(S.eigenstates @ self @ S.eigenstates.dagger()) 
        #     out = self.unitary_transformation(S)
        #     # if self.is_diagonalized():
        #     #     out.eigenstates = S.eigenstates @ self.eigenstates                
                
        # else:
        #     raise ValueError("'direction' can be only 'forward' or 'backward'")
        
        # # this is the same in all the cases
        # if self.is_diagonalized():
        #     out.eigenvalues = copy(self.eigenvalues)
        
        try:
            out.extras = deepcopy(self.extras)
        except:
            pass
        return out#.clean()


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
        
        sym = S[0].clean()

        if not self.commute(sym):
            raise ValueError('You have provided a symmetry operator which does not commute with the operator that you want to diagonalize.')
        
        if not sym.is_diagonalized():
            raise ValueError("The symmetry operator 'S' should have already been diagonalized.")

        w,labels = unique_with_tolerance(sym.eigenvalues)
        
        to_diag:T = self.change_basis(sym,direction="forward")
        for n in range(1,len(S)):
            S[n] = S[n].change_basis(sym,direction="forward")

        #  `to_diag` will have the same block form of `sym`
        # and it will be usefull in `to_diag.clean_block_form` to remove 
        # sporious off diagonal elements (numerical noise)
        # to_diag.blocks = labels
        # to_diag.n_blocks = len(np.unique(labels))
    
        
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
        
        self.eigenvalues = to_diag.eigenvalues.copy()
        self.eigenstates = to_diag.eigenstates.copy()
        self.extras      = to_diag.extras.copy()

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
    
    def thermal_average(self:T,temperatures:np.ndarray,operator:T)->np.ndarray:
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
        return quantum_thermal_average_value(T=temperatures,E=self.eigenvalues,Op=operator,Psi=self.eigenstates)
    
    # @property
    # def eigenstates(self)->Self:
    #     return super().eigenstates
    
    # @eigenstates.setter
    # def eigenstates(self,value:Self):
    #     # if self.extras.get("memory-saving",False) and self.eigenstates is not None:
    #     #     raise ValueError("You can not modify the eigenstates of an Operator in memory-saving mode.")
    #     self.extras["memory-saving"] = False
    #     if "memory-saving-A" in self.extras:
    #         del self.extras["memory-saving-A"]
    #     if "memory-saving-B" in self.extras:
    #         del self.extras["memory-saving-B"]
    #     self.eigenstates = value  # call parent setter
            
    def save(self: T, file: str) -> None:
        """
        Saves a Matrix object to a file.

        Parameters
        ----------
        file : str
            The name of the file to save the matrix to.

        Returns
        -------
        None
            Saves the matrix to the specified file.
        """
        if self.extras.get("memory-saving",False):
            if "memory-saving-A" not in self.extras or "memory-saving-B" not in self.extras:
                raise ValueError("This is a coding error.")
            to_save = self.copy()
            to_save.eigenstates = None
            super(Operator,to_save).save(file)
        else:
            super().save(file)

    @classmethod
    def load(cls: Type[T], file: str) -> T:
        """
        Loads a Matrix object from a file.

        Parameters
        ----------
        file : str
            The file path to load the object from.

        Returns
        -------
        T
            The loaded Matrix object.
        """
        obj = super().load(file)
        if obj.extras.get("memory-saving",False):
            obj.eigenstates = obj.extras["memory-saving-A"] @ obj.extras["memory-saving-B"]
            # obj.eigenstates = obj.extras.pop("memory-saving-A",None) @ obj.extras.pop("memory-saving-B",None)
            # obj.extras["memory-saving"] = False
        return cls(obj)
         
def test_operator_save_load(tmp_path):
    """
    Test that saving and loading a Matrix preserves its data.
    """
    from quantumsparse.tools.bookkeeping import TOLERANCE
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
    from pathlib import Path
    tmp_path = Path(".")
    test_operator_save_load(tmp_path)