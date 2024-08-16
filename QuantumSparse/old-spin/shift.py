from QuantumSparse.operator import Operator
import numpy as np

def shift(ops:Operator)->Operator:

    basis = np.asarray(ops.basis)
    D = ops.empty()
    left = None

    for c,right in enumerate(basis):
        if left is None:
            left = right.copy()
        left[0]  = right[-1]
        left[1:] = right[:-1]
        # = np.concatenate((right[-1].reshape((1)), right[:-1]))
        r = np.where(np.all(basis == left, axis=1))
        D[r,c] = 1
    return D

    D = ops.empty()


    print(ops.degeneracies)
    N = len(ops.degeneracies)

    # D = None #ops.empty()
    D = None #ops.empty()
    for n in range(N): # cycle over all the sites
        m = n+1 if n<N-1 else 0
        deg = ops.degeneracies[n]
        P = None # plus
        M = None # minus
        for i in range(deg-1): # cycle over the states
            if i == 0 :
                P = ops.Sp[m] 
                M = ops.Sm[n] 
            else:
                P = P @ ops.Sp[m]  
                M = M @ ops.Sm[n] 
        # D += P @ M
        if D is not None:
            D = D + P @ M
        else:
            D = P @ M

    # D.visualize()
    w,f = D.diagonalize(method="dense")

    phi = 1 #2*np.pi/N
    ew = np.exp(1.j*w*phi)
    W = D.empty()
    W.setdiag(ew)


    
    
    E = f.dagger() @ W @ f # np.linalg.inv(f.todense())

    # E[ E < 0.5 ] = 0
    # E[ E > 0.5 ] = 1
    E.visualize(False)

    return E

