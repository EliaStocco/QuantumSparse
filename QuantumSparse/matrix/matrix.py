from scipy.linalg import eigh, eig
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# from scipy.sparse import issparse, diags, hstack
from copy import copy, deepcopy
from scipy.sparse import bmat
import numpy as np
# from QuantumSparse.tools.optimize import jit
from QuantumSparse.errors import ImplErr
from QuantumSparse.tools import first_larger_than_N, get_deep_size
from typing import TypeVar, Union, Type, List
import pickle
from dataclasses import dataclass, field
import concurrent.futures
    
T = TypeVar('T',bound="Matrix") 
dtype = csr_matrix
NoJacobi = 8

USE_COPY = True
def my_copy(x):
    if USE_COPY:
        return copy(x)
    else:
        return x

# class get_class(type):
#     def __new__(cls, name, bases, attrs):
#         # Iterate through the attributes of the class
#         if issubclass(dtype, sparse.spmatrix) :
#             attrs["module"] = sparse
#         else :
#             raise ImplErr
#        
#         bases = bases + (dtype,)
#
#         # Create the class using the modified attributes
#         return super().__new__(cls, name, bases, attrs)
    

# class matrix(metaclass=get_class)
class Matrix(csr_matrix):
    """
    Class to handle matrices in different form, i.e. dense, sparse, with 'numpy', 'scipy', or 'torch'
    """
    
    module = sparse
    
    blocks:list
    n_blocks:int
    eigenvalues:np.ndarray 
    eigenstates:T
    ##NEARLY_DIAG## nearly_diag:bool
    is_adjacency:bool

    def __init__(self:T,*argc,**argv):
        """
        Initializes a Matrix object.

        Parameters
        ----------
        *argc : variable number of positional arguments
            Positional arguments to be passed to the parent class.
        **argv : variable number of keyword arguments
            Keyword arguments to be passed to the parent class.

        Returns
        -------
        None
            Initializes the Matrix object and sets its attributes.
        """
        # https://realpython.com/python-super/
        super().__init__(*argc,**argv)

        self.blocks = None
        self.n_blocks = None
        self.eigenvalues = None
        self.eigenstates = None
        ##NEARLY_DIAG## self.nearly_diag = None
        self.is_adjacency = None
        # if is_adjacency:
        #     self.is_adjacency = True
        # else:
        #     self.is_adjacency = self.det_is_adjacency()
    
    def clone(self:T,*argc,**argv)->T:
        """Clone a matrix

        Returns
        ----------
        Matrix
            A new instance of the same matrix
        """
        return type(self)(*argc,**argv)
    
    def copy(self:T)->T:
        """
        Creates a deep copy of the current Matrix object.

        Returns
        -------
        Matrix
            A deep copy of the current Matrix object.
        """
        return deepcopy(self)

    # def save(self:T,file):
    #     """Save a matrix in a file

    #     Parameters
    #     ----------
    #     file : str
    #         The name of the file to save the matrix
    #     """
    #     if Matrix.module is sparse :
    #         sparse.save_npz(file,self)
    #     else :
    #         raise ImplErr
    
    # @staticmethod
    # def load(file):
    #     """
    #     Load a matrix from a file.

    #     Parameters:
    #         file (str): The name of the file to load the matrix from.

    #     Returns:
    #         The loaded matrix if the module is 'sparse', otherwise raises ImplErr.
    #     """
    #     if Matrix.module is sparse :
    #         return sparse.load_npz(file)
    #     else :
    #         raise ImplErr
    
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
        with open(file, 'wb') as f:
            pickle.dump(self, f)

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
        with open(file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def diags(cls,*argc,**argv):
        """
        Creates a diagonal matrix using the module's diags function.

        Parameters
        ----------
        *argc : variable number of positional arguments
            Positional arguments to be passed to the module's diags function.
        **argv : variable number of keyword arguments
            Keyword arguments to be passed to the module's diags function.

        Returns
        -------
        Matrix
            A new Matrix instance representing the diagonal matrix.
        """
        return cls(Matrix.module.diags(*argc,**argv))

    @classmethod
    def kron(cls,*argc,**argv):
        """
        Create a new Matrix instance by taking the Kronecker product of the given matrices.

        Parameters:
            *argc (Matrix or array-like): The matrices to take the Kronecker product of.
            **argv (dict): Additional keyword arguments to pass to the Kronecker product function.

        Returns:
            Matrix: A new Matrix instance representing the Kronecker product of the given matrices.
        """
        return cls(Matrix.module.kron(*argc,**argv))

    @classmethod
    def identity(cls,*argc,**argv):
        """
        Creates an identity matrix using the module's identity function.

        Parameters
        ----------
        *argc : variable number of positional arguments
            Positional arguments to be passed to the module's identity function.
        **argv : variable number of keyword arguments
            Keyword arguments to be passed to the module's identity function.

        Returns
        -------
        Matrix
            A new Matrix instance representing the identity matrix.
        """
        return cls(Matrix.module.identity(*argc,**argv))
    
    def dagger(self:T)->T:
        """
        Returns the Hermitian conjugate (or adjoint) of the matrix.

        The Hermitian conjugate of a matrix is obtained by taking the transpose of the matrix and then taking the complex conjugate of each entry.

        Returns
        -------
        T
            The Hermitian conjugate of the matrix.
        """
        return self.clone(self.conjugate().transpose()) #type(self)(self.conjugate().transpose())

    def is_symmetric(self:T,**argv)->bool:
        """
        Checks if the matrix is symmetric.

        Parameters:
            **argv (dict): Additional keyword arguments.
                tolerance (float): The tolerance for checking symmetry.

        Returns:
            bool: True if the matrix is symmetric, False otherwise.
        """
        if Matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.transpose()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_hermitean(self:T,**argv)->bool:
        """
        Checks if the matrix is Hermitian.

        Parameters:
            **argv (dict): Additional keyword arguments.
                tolerance (float): The tolerance for checking Hermiticity.

        Returns:
            bool: True if the matrix is Hermitian, False otherwise.
        """
        if Matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.dagger()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_unitary(self:T,**argv)->bool:
        """
        Checks if the matrix is unitary.

        A unitary matrix is a square matrix whose columns and rows are orthonormal vectors.

        Parameters:
            **argv (dict): Additional keyword arguments.
                tolerance (float): The tolerance for checking unitarity.

        Returns:
            bool: True if the matrix is unitary, False otherwise.

        Raises:
            ImplErr: If the matrix module is not sparse.
        """
        if Matrix.module is sparse :
            tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self @ self.dagger() - self.identity(len(self)) ).norm() < tolerance
        else :
            raise ImplErr
        
    def det_is_adjacency(self:T):
        """
        Checks if the matrix is an adjacency matrix.

        Returns:
            bool: True if the matrix is an adjacency matrix, False otherwise.
        """
        if self.is_adjacency is None:
            self.is_adjacency = self == self.adjacency()
            
        return self.is_adjacency

    def diagonalized(self:T)->bool:
        """
        Check if the matrix is diagonalized.

        Returns:
            bool: True if the matrix is diagonalized, False otherwise.
        """
        return self.eigenvalues is not None and self.eigenstates  is not None

    @staticmethod
    def anticommutator(A:T,B:T)->T:
        """
        Computes the anticommutator of two matrices A and B.

        Args:
            A (T): The first matrix.
            B (T): The second matrix.

        Returns:
            T: The anticommutator of A and B.
        """
        return A @ B + B @ A 
    
    @staticmethod
    def commutator(A:T,B:T)->T:
        """
        Computes the commutator of two matrices A and B.

        The commutator is defined as the difference between the matrix product AB and BA.

        Parameters:
            A (T): The first matrix.
            B (T): The second matrix.

        Returns:
            T: The commutator of A and B.
        """
        return A @ B - B @ A 
    
    def commute(self:T,A,tol=1e-6)->bool:
        """
        Checks if the matrix is commutative with another matrix A within a given tolerance.

        Args:
            A (Matrix): The matrix to check commutativity with.
            tol (float, optional): The tolerance for commutativity. Defaults to 1e-6.

        Returns:
            bool: True if the matrix is commutative with A within the given tolerance, False otherwise.
        """
        return Matrix.commutator(self,A).norm() < tol

    # @staticmethod 
    def norm(self:T)->float:
        """
        Computes the Euclidean norm (magnitude) of the matrix.

        Parameters:
            self (Matrix): The matrix to compute the norm of.

        Returns:
            float: The Euclidean norm of the matrix.

        Raises:
            ImplErr: If the matrix module is not sparse.
        """
        if Matrix.module is sparse :
            return sparse.linalg.norm(self)
        else :
            raise ImplErr

    def adjacency(self:T)->T:
        """
        Computes the adjacency matrix of the given matrix.

        The adjacency matrix is a binary matrix where the entry at row i and column j is 1 if there is an edge between vertices i and j, and 0 otherwise.

        Parameters:
            self (Matrix): The matrix to compute the adjacency matrix of.

        Returns:
            T: The adjacency matrix of the given matrix.

        Raises:
            ImplErr: If the matrix module is not sparse.
        """
        if Matrix.module is sparse :
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
    
    def sparsity(self:T)->float:
        """
        Computes the sparsity of the given matrix.

        The sparsity of a matrix is the ratio of the number of non-zero elements to the total number of elements in the matrix.

        Parameters:
            self (Matrix): The matrix to compute the sparsity of.

        Returns:
            float: The sparsity of the given matrix.

        Raises:
            ImplErr: If the matrix module is not sparse.
        """
        if Matrix.module is sparse :
            rows, cols = self.nonzero()
            shape = self.shape
            return float(len(rows)) / float(shape[0]*shape[1])
        else :
            raise ImplErr
        # adjacency = self.adjacency()
        # v = adjacency.flatten()
        # return float(matrix.norm(v)) / float(len(v))

    def visualize(self:T,adjacency=True,tab='tab10',cb=True,file=None)->None:
        """
        Visualizes a matrix using matplotlib.

        Parameters:
            adjacency (bool): Whether to visualize the adjacency matrix. Defaults to True.
            tab (str): The colormap to use. Defaults to 'tab10'.
            cb (bool): Whether to count blocks before visualizing. Defaults to True.
            file (str): The file to save the visualization to. If None, the visualization will be displayed. Defaults to None.

        Returns:
            None
        """
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
    
    def csr_memory_usage(self):
        """Return the memory usage of the csr_matrix part of the object."""
        # Accessing the csr_matrix internal components
        data_mem = self.data.nbytes
        indices_mem = self.indices.nbytes
        indptr_mem = self.indptr.nbytes
        
        total_mem = data_mem + indices_mem + indptr_mem
        return total_mem
    
    def __repr__(self:T)->str:
        """
        Returns a string representation of the matrix object, including its type, shape, 
        sparsity, number of elements, norms, and symmetry properties.
        """
        
        string  = "{:>12s}: {} bytes\n".format('memory (csr)', str(get_deep_size(csr_matrix(self))))
        string += "{:>12s}: {} bytes\n".format('memory (deep)', str(get_deep_size(self)))
        string += "{:>12s}: {}\n".format('type', str(self.data.dtype))
        string += "{:>12s}: {}\n".format('shape', str(self.shape))
        string += "{:>12s}: {:6f}\n".format('sparsity', self.sparsity())
        string += "{:>12s}: {}\n".format('# all', str(self.count("all")))
        string += "{:>12s}: {}\n".format('#  on', str(self.count("diag")))
        string += "{:>12s}: {}\n".format('# off', str(self.count("off")))
        string += "{:>12s}: {:6f}\n".format('norm (all)', self.norm())
        string += "{:>12s}: {:6f}\n".format('norm  (on)', self.as_diagonal().norm())
        string += "{:>12s}: {:6f}\n".format('norm (off)', self.off_diagonal().norm())
        string += "{:>12s}: {}\n".format('hermitean', str(self.is_hermitean()))
        string += "{:>12s}: {}\n".format('symmetric', str(self.is_symmetric()))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.is_unitary)
            timeout_duration = 1
            try:
                # Wait for the result with a timeout
                is_unitary = future.result(timeout=timeout_duration)
            except concurrent.futures.TimeoutError:
                is_unitary = "unknown"  # Return the default value if the task times out
                
        string += "{:>12s}: {}\n".format('unitary', str(is_unitary))
        
        tmp = str(self.n_blocks) if self.n_blocks is not None else "unknown"
        string += "{:>12s}: {}\n".format('n. blocks', tmp)
        
        tmp = "computed" if self.eigenvalues is not None else "unknown"
        string += "{:>12s}: {}\n".format('eigenvalues', tmp)
        tmp = "computed" if self.eigenstates is not None else "unknown"
        string += "{:>12s}: {}\n".format('eigenstates', tmp)

        return string
    
    def __len__(self:T)->int:
        """
        Returns the length of the matrix.

        Returns:
            int: The length of the matrix.

        Raises:
            ValueError: If the matrix is not square.
        """
        M,N = self.shape
        if M != N :
            raise ValueError("matrix is not square: __len__ is not well defined")
        return M
    
    def empty(self:T)->T:
        """
        Creates an empty matrix of the same shape and with the same dtype as the original matrix.
        
        Returns:
            T: A new Matrix instance representing an empty matrix.
        """
        return self.clone(self.shape,dtype=self.dtype) # type(self)(self.shape,dtype=self.dtype)
    
    @staticmethod
    def one_hot(dim:int,i:int,j:int,*argc,**argv)->T:
        """
        Creates a one-hot matrix of the same shape and with the same dtype as the original matrix.
        
        Parameters:
            i (int): The row index of the one-hot element.
            j (int): The column index of the one-hot element.
        
        Returns:
            T: A new Matrix instance representing a one-hot matrix.
        """
        empty = Matrix((dim,dim),*argc,**argv)
        empty[i,j] = 1
        return empty
    
    def as_diagonal(self:T)->T:
        """
        Creates a diagonal matrix from the diagonal elements of the original matrix.
        
        Returns:
            T: A new Matrix instance representing a diagonal matrix.
        """
        diagonal_elements = super().diagonal()
        return type(self).diags(diagonals=diagonal_elements)
    
    def off_diagonal(self:T)->T:
        """
        Creates an off-diagonal matrix from the non-diagonal elements of the original matrix.
        
        Returns:
            T: A new Matrix instance representing an off-diagonal matrix.
        """
        diagonal_elements = super().diagonal() # self.diagonal()
        diagonal_matrix = type(self).diags(diagonals=diagonal_elements)
        return self - diagonal_matrix
    
    def count(self:T,what="all")->int:
        """
        Counts the number of elements of the matrix, either all, diagonal or off-diagonal elements.
        
        Parameters:
            what (str): The type of elements to be counted. Default is "all".
        
        Returns:
            int: The number of elements of the specified type.
        
        Raises:
            ValueError: If 'what' is not one of "all", "diag", or "off".
        """
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
        """
        Creates a matrix from a list of blocks.
        
        Parameters:
            blocks (list): A list of blocks.
        
        Returns:
            T: A new Matrix instance representing the matrix created from the blocks.
        """
        N = len(blocks)
        tmp = np.full((N,N),None,dtype=object)
        for n in range(N):
            tmp[n,n] = blocks[n]
        if Matrix.module is sparse : 
            return cls(sparse.bmat(tmp))
        else :
            raise ValueError("error")

    def count_blocks(self:T, inplace=True):
        """
        Counts the number of blocks in the matrix.

        Parameters:
            inplace (bool): If True, updates the blocks and n_blocks attributes of the matrix. Default is True.

        Returns:
            tuple: A tuple containing the number of connected components and the labels of the connected components.

        Raises:
            ImplErr: If the matrix module is not sparse.
        """
        adjacency = self.adjacency()
        if Matrix.module is sparse:
            n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
            if inplace:
                self.blocks = labels
                self.n_blocks = len(np.unique(labels))
            return n_components, labels
        else:
            raise ImplErr
        
    def block_summary(self):
        """
        Provides a summary of the blocks in the matrix.

        Prints the number of blocks and the dimensions of each subspace.

        Returns:
            np.ndarray: An array containing the dimensions of each subspace.
        """
        nblocks, labels = self.count_blocks()
        print("n. blocks: ",nblocks)
        subspaces = np.unique(labels)
        assert len(subspaces) == nblocks, "Coding error"
        dim = np.zeros(len(subspaces),dtype=int)
        for n,s in enumerate(subspaces):
            dim[n] = np.sum(labels == s)
        print("subspaces dimension: ",dim.tolist())
        return dim

    def mask2submatrix(self:T, mask) -> T:
        """
        Creates a submatrix from the mask.

        Parameters:
            mask (array-like): A boolean mask indicating the rows and columns to include in the submatrix.

        Returns:
            T: A new matrix instance representing the submatrix.
        """
        submatrix = self[mask][:, mask]  #

        # this is needed to restart from a previous calculation
        if self.eigenvalues is not None:
            submatrix.eigenvalues = self.eigenvalues[mask]
        if self.eigenstates is not None:
            submatrix.eigenstates = self.eigenstates[mask][:, mask]
        ##NEARLY_DIAG## if self.nearly_diag is not None:
        ##NEARLY_DIAG##     submatrix.nearly_diag = self.nearly_diag[mask][:, mask]

        return submatrix

    def divide_into_block(self: T, labels: np.ndarray, sort=True) -> T:
        """
        Divides the matrix into blocks.

        Parameters:
            labels (array-like): An array of labels indicating the block each element belongs to.
            sort (bool): If True, sorts the matrix according to the labels. Default is True.

        Returns:
            T: A new matrix instance representing the matrix divided into blocks.
        """
        submatrices = np.full((self.n_blocks, self.n_blocks), None, dtype=object)
        # submatrices = [None]*self.n_blocks
        indeces = np.arange(self.shape[0])
        permutation = np.arange(self.shape[0])

        k = 0
        for n in range(self.n_blocks):
            mask = (labels == n)
            permutation[k:k + len(indeces[mask])] = indeces[mask]
            k += len(indeces[mask])
            # create a submatrix from one block
            submatrices[n,n] = self.mask2submatrix(mask)

        out = self.clone(bmat(submatrices))
        # out = DiagonalBlockMatrix(submatrices)
        if sort:
            reverse_permutation = np.argsort(permutation)
            out = out[reverse_permutation][:, reverse_permutation]
        return out

    # #@jit
    def diagonalize_each_block(self: T, labels: np.ndarray, original: bool, tol: float,
                               max_iter: int) -> Union[np.ndarray, T]:
        """
        Diagonalizes each block of the matrix.

        Parameters:
            labels (array-like): An array of labels indicating the block each element belongs to.
            original (bool): If True, prints the block being diagonalized. Default is True.
            tol (float): The tolerance for convergence. Default is 1e-3.
            max_iter (int): The maximum number of iterations. If -1, there is no maximum. Default is -1.

        Returns:
            tuple: A tuple containing the eigenvalues, eigenstates, and nearly diagonal matrix.

        Raises:
            ValueError: If original is False.
        """
        ##NEARLY_DIAG##submatrices = np.full((self.n_blocks, self.n_blocks), None, dtype=object)
        eigenvalues = np.full(self.n_blocks, None, dtype=object)
        eigenstates = np.full(self.n_blocks, None, dtype=object)

        if original:
            indeces = np.arange(self.shape[0])
            permutation = np.arange(self.shape[0])
            k = 0
            print("\tStarting diagonalization")
        else:
            raise ValueError("some error occurred")

        for n in range(self.n_blocks):
            if original:
                print(f"\t\tdiagonalizing block n. {n}/{self.n_blocks}")

            mask = (labels == n)
            permutation[k:k + len(indeces[mask])] = indeces[mask]
            k += len(indeces[mask])

            # create a submatrix from one block
            submatrix = self.mask2submatrix(mask)

            # diagonalize the block
            ##NEARLY_DIAG##v, f, M = submatrix.eigensolver(original=False, method=method, tol=tol, max_iter=max_iter)
            ##NEARLY_DIAG##submatrices[n, n] = M
            eigenvalues[n], eigenstates[n] = submatrix.eigensolver(original=False, tol=tol, max_iter=max_iter)
            # eigenvalues[n] = v
            # eigenstates[n] = f

        eigenvalues = np.concatenate(eigenvalues)
        
        # Attention:
        # This is extremely inefficient for large matrices
        # because if you are using a symmetry operator
        # when coming back to the original basis 
        # the eigenstates will no longer be in block form.
        eigenstates = Matrix.from_blocks(eigenstates)
        # eigenstates = DiagonalBlockMatrix(eigenstates)

        reverse_permutation = np.argsort(permutation)
        eigenvalues = eigenvalues[reverse_permutation]
        eigenstates = eigenstates[reverse_permutation][:, reverse_permutation]
        ##NEARLY_DIAG## nearly_diagonal = self.clone(bmat(submatrices))[reverse_permutation][:, reverse_permutation]
        # type(self)(bmat(submatrices))
        return eigenvalues, eigenstates##NEARLY_DIAG## , nearly_diagonal
 
    def eigensolver(self:T,original=True,tol:float=1.0e-3,max_iter:int=-1):
        """
        Diagonalize the matrix using the specified method.

        Parameters
        ----------
        original : bool, optional
            If True, the block being diagonalized is printed (default is True).
        tol : float, optional
            The tolerance for the diagonalization process (default is 1.0e-3).
        max_iter : int, optional
            The maximum number of iterations for the diagonalization process (default is -1).

        Returns
        -------
        w : numpy.ndarray
            The eigenvalues of the matrix.
        f : numpy.ndarray
            The eigenstates of the matrix.
        N : Matrix
            The nearly diagonal matrix.
        """
        # if Matrix.module is sparse :

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
        if original : 
            print("\tn_components:",n_components) 
        elif self.n_blocks != 1 :
            raise ValueError("some error occurred")

        if self.n_blocks == 1 :
            # if self.shape[0] < NoJacobi:
            #     method = "dense"
            # if method == "jacobi":
            #     raise ValueError("method 'jacobi' is no longer supported.")
            #     if not self.is_hermitean():
            #         raise ValueError("'matrix' object is not hermitean and then it can not be diagonalized with the Jacobi method")
            #     from QuantumSparse.matrix.jacobi import jacobi
            #     w,f,N = jacobi(self,tol=tol,max_iter=max_iter)
            # elif method == "dense":
            M = np.asarray(self.todense())
            # N = self.empty()
            self.eigenvalues,self.eigenstates = eigh(M) if self.is_hermitean() else eig(M)
            # if self.is_hermitean():
            #     self.eigenvalues,self.eigenstates = eigh(M) if self.is_hermitean() else eig(M)
            # else :
            #     self.eigenvalues,self.eigenstates = eig(M)
            # N = N.astype(w.dtype)
            # N.setdiag(w)
            # pass
            # else :
            #     raise ImplErr
        
        elif self.n_blocks > 1:
            ##NEARLY_DIAG## w,f,N = self.diagonalize_each_block(labels=labels,original=True,method=method,tol=tol,max_iter=max_iter)
            self.eigenvalues,self.eigenstates = self.diagonalize_each_block(labels=labels,original=True,tol=tol,max_iter=max_iter)

        else :
            raise ValueError("error: n. of block should be >= 1")
        
        # self.eigenvalues = my_copy(w)
        # self.eigenstates = my_copy(f)
        # self.eigenvalues = w 
        # self.eigenstates = f
        
        ##NEARLY_DIAG## self.nearly_diag = my_copy(N) if N is not None else None
        
        # if isinstance(self.eigenstates,np.ndarray):
        #     self.eigenstates = Matrix(self.eigenstates)
        # else:
        #     pass
        
        # return w,f##NEARLY_DIAG##,N
        return self.eigenvalues,self.eigenstates
    
    def test_eigensolution(self:T)->T:
        """
        Test the eigensolution and return the norm of the error.

        Returns
        -------
        norm : float
            The norm of the error.
        """
        return self @ self.eigenstates - self.eigenstates @ self.diags(self.eigenvalues)
    
    def sort(self:T,inplace=False)->T:
        
        out = self.copy()
        index = np.argsort(self.eigenvalues)
        out.eigenvalues = self.eigenvalues[index]
        out.eigenstates = self.eigenstates[index][:, index]
        ##NEARLY_DIAG## out.nearly_diag = self.nearly_diag[index][:, index]
        out = self[index][:, index]
        if inplace:
            self = out
        return out
        

    
@dataclass
class DiagonalBlockMatrix(Matrix):
    
    blocks: List[Matrix]
    cumulative_indices:List[int] = field(init=False)
    
    # def __post_init__(self):
    #     assert self.indices.ndim == 2, "error: 'indices' should be a 2D array"
    #     assert self.indices.shape[1] == 2, "error: 'indices.shape[1]' should be of lenght 2"
    #     assert self.indices.shape[0] == len(self.blocks), "error: 'indices.shape[0]' should be equal to 'len(self.blocks)'"
    
    def __post_init__(self):
        for block in self.blocks:
            assert block.shape[0] == block.shape[1], "error: each block should be square"
        self.cumulative_indices = np.cumsum([0]+[block.shape[0] for block in self.blocks]).tolist()
    
    def __getitem__(self, pos):
        """Retrieve an element based on global position (row, col)."""
        row, col = pos
        iR = first_larger_than_N(self.cumulative_indices[1:],row)
        iC = first_larger_than_N(self.cumulative_indices[1:],col)

        if iR is None or iC is None:
            raise IndexError(f"Position ({row}, {col}) is out of bounds in the BlockMatrix")
        elif iR == iC:
            block = self.blocks[iR]
            iR = row - self.cumulative_indices[iR]
            iC = col - self.cumulative_indices[iC]
            return block[iR,iC]
        else: 
            return 0 

    def __setitem__(self, pos, value):
        """Set an element based on global position (row, col)."""
        row, col = pos

        # Find the block that contains the (row, col) position
        for (i, block) in enumerate(self.blocks):
            block_start_row, block_start_col = self.indices[i]
            block_end_row = block_start_row + block.data.shape[0]
            block_end_col = block_start_col + block.data.shape[1]

            # Check if the requested position lies in the current block
            if block_start_row <= row < block_end_row and block_start_col <= col < block_end_col:
                local_row = row - block_start_row
                local_col = col - block_start_col
                block[local_row, local_col] = value
                return

        raise IndexError(f"Position ({row}, {col}) is out of bounds in the BlockMatrix")
    
    def shape(self):
        return self.cumulative_indices[-1], self.cumulative_indices[-1]

    # def shape(self):
    #     """Return the global shape of the BlockMatrix."""
    #     total_rows = max(start_row + block.data.shape[0] for start_row, _ in self.indices)
    #     total_cols = max(start_col + block.data.shape[1] for _, start_col in self.indices)
    #     return total_rows, total_cols
        

def main():
    
    import numpy as np
    from icecream import ic
    
    # Define individual block matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6],[7, 8]])
    C = np.array([[9, 10,4],[11, 12,0],[11, 12,2]])
    null = np.zeros((2,2))
    
    block1 = Matrix(A)
    block2 = Matrix(B)
    block3 = Matrix(C)

    # Create the BlockMatrix
    bm = DiagonalBlockMatrix(blocks=[block1, block2,block3])
    
    test = sparse.bmat([[A,None,None],[None,B,None],[None,None,C]],format="csr")# .toarray()
    ic(test)
    
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            assert test[i,j] == bm[i,j], "error: block matrices do not match"
               
    

if __name__ == "__main__":
    main()