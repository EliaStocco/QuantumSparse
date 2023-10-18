# "operator" class
import numpy as np
from ..matrix import matrix #, identity

class operator(matrix):
    
   
    def __init__(self,*argc,**argv):
        # https://realpython.com/python-super/
        super().__init__(*argc,**argv)

        self.eigenvalues = None
        self.eigenstates = None
        self.permutation = None
        pass
    
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
                #print("\t",i+1,"/",N,end="\r")        
                #iden[i] = sparse.diags(np.full(dim,1,dtype=int),dtype=int)  
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
    
    def diagonalize(self,inplace=False,**argv):

        if not self.is_hermitean():
            raise ValueError("'operator' is not hermitean")
        
        w,f,permutation = super().diagonalize()
        self.eigenvalues = w
        self.eigenstates = f
        self.permutation = permutation

        return self.eigenvalues, self.eigenstates

    
    # @staticmethod
    # def sum(Ops):
    #     """
    #     Parameters
    #     ----------
    #     Ops : np.array of scipy.sparse
    #         array of operator to be summed,
    #         each acting on the system Hilbert space
        
    #     Returns
    #     -------
    #     tot : scipy.sparse
    #         sum of given operator
    #     """
    #     dims = [ Op.shape for Op in Ops ]
    #     boolean = [ dim == dims[0] for dim in dims ]
    #     if not np.all(boolean) :
    #         print("\t\terror in \"sum\" function: not all operator with the same (matrix representation) dimensions")
    #         raise()
    #     tot = 0 
    #     for Op in Ops:
    #         tot += Op
    #     return tot
    
