# compute identity operators
import numpy as np
from scipy import sparse


def identity(NSpin,spin_values,Degeneracies):
    """
    Parameters
    ----------
    NSpin : TYPE
        DESCRIPTION.
    spin_values : TYPE
        DESCRIPTION.
    Degeneracies : TYPE
        DESCRIPTION.

    Returns
    -------
    iden : TYPE
        DESCRIPTION.
    """    
    iden = np.zeros(NSpin,dtype=object)
    for i,s,deg in zip(range(NSpin),spin_values,Degeneracies):
        print("\t",i+1,"/",NSpin,end="\r")        
        iden[i] = sparse.diags(np.full(deg,1,dtype=int),dtype=int)           
    return iden