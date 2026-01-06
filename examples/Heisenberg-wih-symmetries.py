import numpy as np
from quantumsparse.spin import SpinOperators, from_S2_to_S
from quantumsparse.operator import Operator
from quantumsparse.operator import Symmetry, roots_of_unity
from quantumsparse.spin import Heisenberg, anisotropy, spin2dim
from quantumsparse.spin.shift import shift
import pytest

# In quantumsparse/spin/interactions.py you can find:
# - Ising
# - Heisenberg
# - DM
# - anisotropy
# - rhombicity
# - BiQuadraticIsing
# - BiQuadraticHeisenberg

@pytest.mark.parametrize("S,NSpin", [(2,4),(3,3),(5,2)])
def test_Heisenberg_symmetries(S=0.5,NSpin=3,use_symmetries=True):
    
    spin_values = np.full(NSpin,S)

    # construct the spin operators
    SpinOp = SpinOperators(spin_values)
    # unpack the operators
    Sx,Sy,Sz = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz
    
    
    
    #-----------------#
    if use_symmetries:
        D:Symmetry = shift(SpinOp)
        print(repr(D))
        nblocks, _ = D.count_blocks()
        print("\tnblocks:",nblocks)
        D.diagonalize(method="dense") # 'dense' is smuch better than 'jacobi'
        # D.eigenvalues.sort()
        # print(D.eigenvalues)
        l,N = D.energy_levels()
        # print(len(l))
        assert len(l) == NSpin, "wrong number of energy levels"
        ru = np.sort(roots_of_unity(len(spin_values)))
        l  = np.sort(l)    
        assert np.allclose(l,ru), "The eigenvalues should be the roots of the unity."
    

    #-----------------#
    # Heisenberg Hamiltonian
    # as soon as you break some trivial symmetry (which is "visible" in the orinal basis)
    # diagonalizing with symmetries becomes more efficient
    # even considering the time spent to diagonalize the symmetry operator
    # (which can be done once, save to file the results, and load them again the next time).
    
    H = Heisenberg(Sx,Sy,Sz,couplings=[1,2,3]) # + anisotropy(Sz)
    print(repr(H))
    
    nblocks, _ = H.count_blocks()
    print("\tnblocks:",nblocks)    
    
    #-----------------#
    # save an indipendent copy of the Hamiltonian 
    # to check that nothing fishy in going on
    # (numpy arrays can be insidious)
    Hold = Operator(H.copy())
    
    assert np.all(H.data == Hold.data), "the data should be the same"
    assert H is not Hold, "the variables should be independent"
    
    #-----------------#
    if use_symmetries:
        E,Psi = H.diagonalize_with_symmetry(S=[D])
    else:
        E,Psi = H.diagonalize()
    test = H.test_eigensolution().norm()
    print("\ttest: ", test)
    
    #-----------------#
    Hold.eigenstates = Psi
    Hold.eigenvalues = E
    test = Hold.test_eigensolution().norm()
    print("\ttest: ", test)

    #-----------------#
    E = E.real
    E.sort()

    print("\tmin eigenvalue:",E[0])
    print("\tmax eigenvalue:",E[-1])
    E = E-min(E)
    print("\tenergy range:",E[-1]-E[0])
    
    #-----------------#
    H.save("Heisenberg.pickle")
    
import contextlib
import sys
import os
@contextlib.contextmanager
def suppress_output(suppress=True):
    if suppress:
        with open(os.devnull, "w") as fnull:
            sys.stdout.flush()  # Flush the current stdout
            sys.stdout = fnull
            try:
                yield
            finally:
                sys.stdout = sys.__stdout__  # Restore the original stdout
    else:
        yield

if __name__ == "__main__":
    
    import time
    SUPPRESS = False
    start_time = time.time()
    with suppress_output(SUPPRESS):
        test_Heisenberg_symmetries(use_symmetries=True)
    end_time = time.time()
    print("\n\nElapsed time:", end_time - start_time, "seconds")
    
    start_time = time.time()
    with suppress_output(SUPPRESS):
        test_Heisenberg_symmetries(use_symmetries=False)
    end_time = time.time()
    print("\n\nElapsed time:", end_time - start_time, "seconds")
