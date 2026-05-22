from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin import SpinOperators
from quantumsparse.tools.mathematics import roots_of_unity
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

def shift(ops: SpinOperators,diagonalize:bool=True) -> Operator:
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
    from scipy.sparse import lil_matrix
    basis = np.asarray(ops.basis)
    N = len(basis)
    basis_lookup = {tuple(b): i for i, b in enumerate(basis)}
    # D = ops.empty()
    D = lil_matrix((N, N))

    def process_state(c):
        right = basis[c]
        left = np.roll(right, 1)
        r = basis_lookup.get(tuple(left), None)
        if r is not None:
            return (r, c)
        return None

    results = [process_state(c) for c in range(N)]

    for result in results:
        if result is not None:
            r, c = result
            D[r, c] = 1 # this is a gauge choice, since it could be phase
            
    S = Symmetry(D)
    
    if diagonalize:

        # --- 1. build permutation ---
        perm = np.full(N, -1, dtype=int)

        for c in range(N):
            right = basis[c]
            left = np.roll(right, 1)
            r = basis_lookup.get(tuple(left))
            if r is not None:
                perm[c] = r

        # --- 2. cycle decomposition ---
        visited = np.zeros(N, dtype=bool)

        eigenvalues = []
        eigenstates = []

        for start in range(N):
            if visited[start]:
                continue

            # build orbit
            cycle = []
            x = start

            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = perm[x]

            L = len(cycle)

            # --- 3. Fourier diagonalization on cycle ---
            for k in range(L):
                vec = np.zeros(N, dtype=complex)

                phase = np.exp(-2j * np.pi * k * np.arange(L) / L)

                for n, state in enumerate(cycle):
                    vec[state] = phase[n] / np.sqrt(L)

                eigenstates.append(vec)
                eigenvalues.append(np.exp(2j * np.pi * k / L))

        eigenvalues = np.array(eigenvalues)
        eigenstates = np.array(eigenstates).T   # columns = eigenvectors
        
        S.eigenvalues = eigenvalues
        S.eigenstates = Operator(eigenstates)
        
        # --- BLOCK STRUCTURE (replacement for count_blocks) ---

        visited = np.zeros(N, dtype=bool)
        labels = np.empty(N, dtype=int)

        block_id = 0

        for i in range(N):
            if visited[i]:
                continue

            x = i

            while not visited[x]:
                visited[x] = True
                labels[x] = block_id
                x = perm[x]

            block_id += 1

        n_blocks = block_id

        # store exactly like count_blocks(inplace=True)
        S.blocks = labels
        S.n_blocks = n_blocks

    return S