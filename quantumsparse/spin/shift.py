from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin import SpinOperators
from quantumsparse.tools.mathematics import roots_of_unity
import numpy as np

from joblib import Parallel, delayed
import numpy as np

def shift_foundamental(N:int):
    T = np.zeros((N,N))
    for n in range(N-1):
        T[n,n+1] = 1
    T[N-1,0] = 1
    T = Symmetry(T)
    w = roots_of_unity(N)
    r = np.linspace(0,1,N,endpoint=False)
    k = np.linspace(0,1,N,endpoint=False)
    f = np.exp(1.j*2*np.pi*np.outer(r,k)*N)
    test = T.set_eigen(w,f)
    T.count_blocks()
    assert test.norm() < 1e-8, "error"
    return T

def shift(ops: SpinOperators, parallel: bool = True) -> Operator:
    """
    Compute the shift/translation operator for a spin system.

    Parameters
    ----------
    ops : SpinOperators
        The operator whose basis is to be shifted.
    parallel : bool, optional
        Whether to run in parallel (default is True).

    Returns
    -------
    Operator
        The shift/translation operator for the spin system.
    """
    basis = np.asarray(ops.basis)
    N = len(basis)
    basis_lookup = {tuple(b): i for i, b in enumerate(basis)}
    D = ops.empty()

    def process_state(c):
        right = basis[c]
        left = np.roll(right, 1)
        r = basis_lookup.get(tuple(left), None)
        if r is not None:
            return (r, c)
        return None

    if parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_state)(c) for c in range(N)
        )
    else:
        results = [process_state(c) for c in range(N)]

    for result in results:
        if result is not None:
            r, c = result
            D[r, c] = 1 # this is a gauge choice, since it could be phase

    return Symmetry(D)


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

