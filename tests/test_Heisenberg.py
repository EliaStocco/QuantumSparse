import numpy as np
import pytest
from quantumsparse.spin import SpinOperators, Heisenberg
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift

@pytest.mark.parametrize("N, S", [
    (2, 0.5),
    (3, 0.5),
    (4, 0.5),
    (4, 1.0),
])
def test_heisenberg_hamiltonian(N: int, S: float) -> Operator:
    """
    Build a Heisenberg Hamiltonian for a ring of N spins of spin-S.

    Args:
        N (int): Number of spin sites.
        S (float): Spin value for each site.

    Returns:
        Operator: Heisenberg Hamiltonian as a sparse operator.
    """
    spin_values = np.full(N, S)
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz

    # Construct Heisenberg Hamiltonian manually
    H_manual:Operator = sum(Sx[i] @ Sx[(i + 1) % N] for i in range(N))
    H_manual += sum(Sy[i] @ Sy[(i + 1) % N] for i in range(N))
    H_manual += sum(Sz[i] @ Sz[(i + 1) % N] for i in range(N))
    # H_manual = Operator(H_manual)

    # Compare with library's Heisenberg Hamiltonian
    H_lib = Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz)
    assert np.allclose(H_lib.todense(), H_manual.todense()), "Mismatch in Heisenberg Hamiltonian construction"


pytest.mark.parametrize("S,NSpin", [(2,4),(3,3),(5,2)])
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