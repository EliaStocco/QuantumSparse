from scipy.linalg import eigh, eig
from scipy import sparse
from scipy.sparse import csr_matrix, csr_array
from scipy.sparse.csgraph import connected_components
# from scipy.sparse import issparse, diags, hstack
from copy import copy, deepcopy
from scipy.sparse import bmat
import numpy as np
# from quantumsparse.tools.optimize import jit
from quantumsparse.bookkeeping import ImplErr, scalar
from quantumsparse.tools import first_larger_than_N, get_deep_size
from typing import TypeVar, Union, Type, List, Dict, Any, Optional
import pickle
from dataclasses import dataclass, field
import concurrent.futures
from tqdm import tqdm
    
T = TypeVar('T',bound="Matrix") 
dtype = csr_matrix
NoJacobi = 8
TOLERANCE = 1e-10
NOISE = 1e-12

USE_COPY = True
def my_copy(x):
    if USE_COPY:
        return copy(x)
    else:
        return x

class State(csr_array):
    
    
    @staticmethod
    def one_hot(dim:int,i:int,*argc,**argv)->T:
        """
        Creates a one-hot matrix of the same shape and with the same dtype as the original matrix.
        
        Parameters:
            i (int): The row index of the one-hot element.
            j (int): The column index of the one-hot element.
        
        Returns:
            T: A new Matrix instance representing a one-hot matrix.
        """
        # Create an empty sparse array
        empty = State((dim,), *argc, **argv)
        # Set the one-hot element
        empty[i] = 1
        return empty
    
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
    extras:Dict[str,Any]
    
    # def __new__(cls, arg1=None, shape=None, dtype=None, copy=False):
    #     # Let scipy create the object safely
    #     obj = super().__new__(cls)
    #     super().__init__(obj, arg1, shape=shape, dtype=dtype, copy=copy)
    #     return obj

    def __init__(self: T, *argc, **argv) -> None:
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
        super().__init__(*argc, **argv)
        self.blocks: Optional[List] = None
        self.n_blocks: Optional[int] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenstates: Optional[T] = None
        self.nearly_diag: Optional[bool] = None
        self.is_adjacency: Optional[bool] = None
        self.extras: Dict[str, Any] = {}
    
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

    def kronecker(self:T,A:T)->T:
        """
        Create a new Matrix instance by taking the Kronecker product of the current matrix with the given matrices.

        Parameters:
            *argc (Matrix or array-like): The matrices to take the Kronecker product of.
            **argv (dict): Additional keyword arguments to pass to the Kronecker product function.

        Returns:
            T: A new Matrix instance representing the Kronecker product of the current matrix and the given matrices.
        """
        new = type(self).kron(self,A)
        if self.is_diagonalized() and A.is_diagonalized():
            a = type(self).diags(self.eigenvalues)
            b = type(A).diags(A.eigenvalues)
            w = type(self).kron(a,b)
            new.eigenvalues = w.diagonal()
            new.eigenstates = type(self).kron(self.eigenstates,A.eigenstates)
        return new
    
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
    
    def iden(self:T)->T:
        assert self.shape[0] == self.shape[1], "The matrix must be square to return the identity matrix."
        return type(self).identity(self.shape[0])
    
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
    
    def inv(self:T)->T:
        """
        Computes the inverse of the matrix.

        Returns
        -------
        T
            The inverse of the matrix.

        Raises
        ------
        ImplErr
            If the matrix module is not sparse.
        """
        if self.is_unitary():
            return self.dagger()
        else:
            if Matrix.module is sparse :
                return self.clone(sparse.linalg.inv(self))
            else:
                raise ImplErr

    def is_symmetric(self:T,tolerance=TOLERANCE)->bool:
        """
        Checks if the matrix is symmetric.

        Parameters:
            **argv (dict): Additional keyword arguments.
                tolerance (float): The tolerance for checking symmetry.

        Returns:
            bool: True if the matrix is symmetric, False otherwise.
        """
        if Matrix.module is sparse :
            # tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.transpose()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_hermitean(self:T,tolerance=TOLERANCE)->bool:
        """
        Checks if the matrix is Hermitian.

        Parameters:
            **argv (dict): Additional keyword arguments.
                tolerance (float): The tolerance for checking Hermiticity.

        Returns:
            bool: True if the matrix is Hermitian, False otherwise.
        """
        if Matrix.module is sparse :
            # tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self - self.dagger()).norm() < tolerance
        else :
            raise ImplErr
        
    def is_unitary(self:T,tolerance=TOLERANCE)->bool:
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
             #tolerance = 1e-10 if "tolerance" not in argv else argv["tolerance"]
            return (self @ self.dagger() - self.identity(len(self)) ).norm() < tolerance
        else :
            raise ImplErr
        
    def is_diagonalized(self:T)->bool:
        """
        Check if the matrix is diagonalized.

        Returns:
            bool: True if the matrix is diagonalized, False otherwise.
        """
        return self.eigenvalues is not None and self.eigenstates  is not None
    
    def is_diagonal(self:T,tolerance=TOLERANCE)->bool:
        test = self - self.as_diagonal()
        return test.norm() < tolerance
    
    def det_is_adjacency(self:T):
        """
        Checks if the matrix is an adjacency matrix.

        Returns:
            bool: True if the matrix is an adjacency matrix, False otherwise.
        """
        if self.is_adjacency is None:
            self.is_adjacency = self == self.adjacency()
            
        return self.is_adjacency

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
        
        string  = "{:>14s}: {} bytes\n".format('memory (csr)', str(get_deep_size(csr_matrix(self))))
        string += "{:>14s}: {} bytes\n".format('memory (deep)', str(get_deep_size(self)))
        string += "{:>14s}: {}\n".format('type', str(self.data.dtype))
        string += "{:>14s}: {}\n".format('shape', str(self.shape))
        string += "{:>14s}: {:6f}\n".format('sparsity', self.sparsity())
        string += "{:>14s}: {}\n".format('# all', str(self.count("all")))
        string += "{:>14s}: {}\n".format('#  on', str(self.count("diag")))
        string += "{:>14s}: {}\n".format('# off', str(self.count("off")))
        string += "{:>14s}: {:6f}\n".format('norm (all)', self.norm())
        string += "{:>14s}: {:6f}\n".format('norm  (on)', self.as_diagonal().norm())
        string += "{:>14s}: {:6f}\n".format('norm (off)', self.off_diagonal().norm())
        string += "{:>14s}: {}\n".format('hermitean', str(self.is_hermitean()))
        string += "{:>14s}: {}\n".format('symmetric', str(self.is_symmetric()))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.is_unitary)
            timeout_duration = 1
            try:
                # Wait for the result with a timeout
                is_unitary = future.result(timeout=timeout_duration)
            except concurrent.futures.TimeoutError:
                is_unitary = "unknown"  # Return the default value if the task times out
                
        string += "{:>14s}: {}\n".format('unitary', str(is_unitary))
        
        tmp = str(self.n_blocks) if self.n_blocks is not None else "unknown"
        string += "{:>14s}: {}\n".format('n. blocks', tmp)
        
        tmp = "computed" if self.eigenvalues is not None else "unknown"
        string += "{:>14s}: {}\n".format('eigenvalues', tmp)
        tmp = "computed" if self.eigenstates is not None else "unknown"
        string += "{:>14s}: {}\n".format('eigenstates', tmp)

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
    
    @classmethod
    def one_hot(cls,dim:int,i:int,j:int,*argc,**argv)->T:
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
        return cls(empty)
    
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

    def divide_into_blocks(self:T, labels: Optional[np.ndarray]=None) -> T:
        """
        Divides a matrix into blocks based on the given labels.

        Parameters:
            labels (Optional[np.ndarray]): The labels of the blocks. If None, the blocks are determined by the matrix's connected components.

        Returns:
            T: A tuple containing an array of submatrices, where each submatrix corresponds to a block, and a permutation array that maps the original indices to the block indices.
        """
        
        if labels is None:
            n_blocks,labels = self.count_blocks(inplace=False)
        else:
            n_blocks = len(np.unique(labels))
            
        submatrices = np.full((n_blocks, n_blocks), None, dtype=object)
        indeces = np.arange(self.shape[0])
        permutation = np.arange(self.shape[0])

        k = 0
        for n in range(n_blocks):
            mask = (labels == n)
            permutation[k:k + len(indeces[mask])] = indeces[mask]
            k += len(indeces[mask])
            # create a submatrix from one block
            submatrices[n,n] = self.mask2submatrix(mask)
        
        # a = self[permutation,:][:,permutation]
        # b = bmat(submatrices)
        # c = a - b
        # assert np.allclose(c.norm(),0)
            
        return submatrices, permutation, np.argsort(permutation), labels
            
    def clean_block_form(self: T, labels: np.ndarray, sort=True) -> T:
        submatrices, permutation, reverse_permutation, _ = self.divide_into_blocks(labels)
        out = self.clone(bmat(submatrices))
        if sort:
            out = out[reverse_permutation][:, reverse_permutation]
        return out

    # #@jit
    def diagonalize_each_block(self: T, labels: np.ndarray, original: bool, tol: float,
                               max_iter: int,**argv) -> Union[np.ndarray, T]:
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
            # print("\tStarting diagonalization")
        else:
            raise ValueError("some error occurred")

        enable_tqdm = argv["tqdm"] if "tqdm" in argv else True
        argv["tqdm"] = False
        for n in tqdm(range(self.n_blocks), disable=not enable_tqdm, desc="Diagonalizing blocks", unit="block"):

            # if original:
            # print(f"\t\tdiagonalizing block n. {n}/{self.n_blocks}")

            mask = (labels == n)
            permutation[k:k + len(indeces[mask])] = indeces[mask]
            k += len(indeces[mask])

            # create a submatrix from one block
            submatrix = self.mask2submatrix(mask)

            # diagonalize the block
            ##NEARLY_DIAG##v, f, M = submatrix.eigensolver(original=False, method=method, tol=tol, max_iter=max_iter)
            ##NEARLY_DIAG##submatrices[n, n] = M
            eigenvalues[n], eigenstates[n] = submatrix.eigensolver(original=False, tol=tol, max_iter=max_iter,**argv)
        
        if "blockeigenstates2extras" in argv and argv["blockeigenstates2extras"]:
            self.extras["blockeigenstates"]    = deepcopy(eigenstates)
            self.extras["blockeigenvalues"]    = deepcopy(eigenvalues)
                    
        # Attention:
        # This is extremely inefficient for large matrices
        # because if you are using a symmetry operator
        # when coming back to the original basis 
        # the eigenstates will no longer be in block form.
        eigenstates = Matrix.from_blocks(eigenstates)
        # eigenstates = DiagonalBlockMatrix(eigenstates)
        eigenvalues = np.concatenate(eigenvalues)
        
        reverse_permutation = np.argsort(permutation)
        eigenvalues = eigenvalues[reverse_permutation]
        eigenstates = eigenstates[reverse_permutation][:, reverse_permutation]
        
        if "blockeigenstates2extras" in argv and argv["blockeigenstates2extras"]:
            # self.extras["blockeigenstates"]    = eigenstates
            self.extras["permutation"]         = permutation
            self.extras["reverse_permutation"] = reverse_permutation
            
        ##NEARLY_DIAG## nearly_diagonal = self.clone(bmat(submatrices))[reverse_permutation][:, reverse_permutation]
        # type(self)(bmat(submatrices))
        return eigenvalues, eigenstates##NEARLY_DIAG## , nearly_diagonal
 
    def eigensolver(self:T,original=True,tol:float=1.0e-3,max_iter:int=-1,**argv):
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

        #############################
        # |-------------------------|
        # |          |   original   |
        # | n_blocks | True | False |
        # |----------|--------------|
        # |    =1    |  ok  |  yes  |
        # |    >1    |  ok  | error |
        # |-------------------------|
        
        if self.is_diagonal():
            self.eigenvalues = self.diagonal()
            self.eigenstates = self.iden()
            return self.eigenvalues, self.eigenstates

        n_components, labels = self.count_blocks(inplace=True)               
        if original : 
            pass
            # print("\tn_components:",n_components) 
        elif self.n_blocks != 1 :
            raise ValueError("some error occurred")

        if self.n_blocks == 1 :
            M = np.asarray(self.todense())
            self.eigenvalues,self.eigenstates = eigh(M) if self.is_hermitean() else eig(M)

        elif self.n_blocks > 1:
            ##NEARLY_DIAG## w,f,N = self.diagonalize_each_block(labels=labels,original=True,method=method,tol=tol,max_iter=max_iter)
            self.eigenvalues,self.eigenstates = self.diagonalize_each_block(labels=labels,original=True,tol=tol,max_iter=max_iter,**argv)

        else :
            raise ValueError("error: n. of block should be >= 1")
        
        ##NEARLY_DIAG## self.nearly_diag = my_copy(N) if N is not None else None
        # return w,f##NEARLY_DIAG##,N
        if isinstance(self.eigenstates,np.ndarray):
            self.eigenstates = Matrix(self.eigenstates)
        return self.eigenvalues,self.eigenstates
    
    def test_eigensolution(self:T)->T:
        """
        Test the eigensolution and return the norm of the error.

        Returns
        -------
        norm : float
            The norm of the error.
        """
        assert self.is_diagonalized(), "The matrix should be already diagonalized."
        if isinstance(self.eigenstates,np.ndarray):
            eigvecs_norm = np.linalg.norm(self.eigenstates,axis=0)
        else:
            eigvecs_norm = np.asarray([ self.eigenstates[:,n].norm() for n in range(self.eigenstates.shape[0])])
        assert np.allclose(eigvecs_norm,1), "Eigenstates should be normalized"
        return Matrix(self @ self.eigenstates - self.eigenstates @ self.diags(self.eigenvalues))
    
    def diagonalize(self:T,restart=False,tol:float=1.0e-3,max_iter:int=-1,**argv):
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
        
        ##NEARLY_DIAG##w,f,_ = super().eigensolver(method=method,original=True,tol=tol,max_iter=max_iter)
        self.eigenvalues,self.eigenstates = self.eigensolver(original=True,tol=tol,max_iter=max_iter,**argv)
        
        self.eigenstates.n_blocks = self.n_blocks
        self.eigenstates.blocks = self.blocks
        
        return self.eigenvalues,self.eigenstates

    def sort(self:T,inplace=False)->T:
        """Sort the eigenvalues, and the eigenvectors acoordingly."""
        out = self.copy()
        index = np.argsort(self.eigenvalues)
        out.eigenvalues = self.eigenvalues[index]
        out.eigenstates = self.eigenstates[index][:, index]
        ##NEARLY_DIAG## out.nearly_diag = self.nearly_diag[index][:, index]
        out = self[index][:, index]
        if inplace:
            self = out
        return out
    
    def __truediv__(self, other):
        result = super().__truediv__(other)
        return self.__class__(result)

    def __itruediv__(self, other):
        result = super().__truediv__(other)
        return self.__class__(result)

    def normalize_eigenvecs(self:T):
        """Normalize inplace the eigenvectors."""
        eigvecs_norm = self.eigenstates.column_norm()
        self.eigenstates /= eigvecs_norm
        eigvecs_norm = self.eigenstates.column_norm()
        assert np.allclose(eigvecs_norm,1), "Eigenstates should be normalized"
        
    def exp(self:T,alpha,method:str="qs",diag_inplace:bool=True,tol:float=1e-8,*argv,**kwargs)->T:
        """Exponential of a matrix via diagonalization and exponentiation of its eigenvalues."""
        if method == "scipy":
            from scipy.sparse.linalg import expm
            return type(self)(expm(alpha*self, *argv, **kwargs))
        elif method == "qs":
            func = lambda x: np.exp(x*alpha)
            return self._eigenvalues_wise_operation(func,diag_inplace=diag_inplace,tol=tol,*argv,**kwargs)
        elif method == "test":
            a = self.exp(alpha,"scipy",diag_inplace=diag_inplace,tol=tol,*argv,**kwargs)
            b = self.exp(alpha,"qs",diag_inplace=diag_inplace,tol=tol,*argv,**kwargs)
            assert (a-b).norm() < tol, "error: expm and qs do not match"
        else:
            raise ValueError(f"Unknown method '{method}' for matrix exponential.")
    
    def ln(self:T)->T:
        """Natural logarithm of a matrix computed via diagonalization and ln of its eigenvalues."""
        func = lambda x: np.log(x)
        return self._eigenvalues_wise_operation(func)
    
    def _eigenvalues_wise_operation(self,func,diag_inplace:bool=True,tol:float=1e-8,*argv,**kwargs):
        """General function to apply function to a matrix via diagonalization, e.g. exp, ln, sqrt."""
        if diag_inplace:
            if not self.is_diagonalized():
                self.diagonalize(*argv,**kwargs)
            new = self.copy()
        else:
            new = self.copy()
            if not new.is_diagonalized():
                new.diagonalize(*argv,**kwargs)
        eigenvalues = func(new.eigenvalues)
        eigenstates:Matrix = new.eigenstates
        new = Matrix(eigenstates@Matrix.diags(eigenvalues)@eigenstates.inv())
        new.eigenvalues = eigenvalues
        new.eigenstates = eigenstates
        # test = new.test_eigensolution()
        # assert test.norm() < tol, "error"
        return new.clean()
    
    def column_norm(self:T)->np.ndarray:
        """
        Computes the column norms of the matrix.

        Returns
        -------
        np.ndarray
            An array containing the norms of each column.
        """
        return np.asarray([ self[:,n].norm() for n in range(self.shape[1])])
    
    def unitary_transformation(self:T,U:T)->T:
        """
        Applies a unitary transformation to the matrix.

        Returns
        -------
        T
            The transformed matrix.
        """
        assert U.is_unitary(), "'U' should be unitary"
        new = U @ self @ U.dagger()
        new = type(self)(new)
        assert type(new) == type(self), "error: 'new' should be of the same type as 'self'"
        if self.is_diagonalized():
            new.eigenvalues = self.eigenvalues
            new.eigenstates = U @ self.eigenstates
        return new.clean()
        
    def clean(self:T,noise=NOISE):
        """
        Removes elements in the matrix with absolute value below `noise`.

        Parameters:
        - noise: float, threshold below which values are considered noise
        """
        mask = np.abs(self.data) >= noise

        self.data = self.data[mask]
        self.indices = self.indices[mask]

        # Reconstruct indptr safely
        row_counts = np.diff(self.indptr)
        kept_counts = np.add.reduceat(mask, np.cumsum(np.r_[0, row_counts[:-1]]))

        # Ensure the correct dtype for indptr (usually int32)
        new_indptr = np.empty_like(self.indptr)
        np.cumsum(np.r_[0, kept_counts], out=new_indptr)

        self.indptr = new_indptr

        # Ensure correct dtype of data and indices if needed
        self.data = self.data.astype(self.data.dtype, copy=False)
        self.indices = self.indices.astype(self.indices.dtype, copy=False)

        self.eliminate_zeros()
        return self
        
        
        

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
    # from icecream import ic
    
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
    # ic(test)
    
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            assert test[i,j] == bm[i,j], "error: block matrices do not match"
               
    

if __name__ == "__main__":
    main()