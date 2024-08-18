import numpy as np
from QuantumSparse.spin import SpinOperators, from_S2_to_S
from QuantumSparse.operator import Operator
from QuantumSparse.operator import Symmetry, roots_of_unity
from QuantumSparse.spin import Heisenberg
from QuantumSparse.spin.shift import shift
import pytest

# In QuantumSparse/spin/interactions.py you can find:
# - Ising
# - Heisenberg
# - DM
# - anisotropy
# - rhombicity
# - BiQuadraticIsing
# - BiQuadraticHeisenberg

@pytest.mark.parametrize("S,NSpin", [(2,4),(3,3),(5,2)])
def test_Ising_symmetries(S=1,NSpin=8):
    
    spin_values = np.full(NSpin,S)

    # construct the spin operators
    SpinOp = SpinOperators(spin_values)
    # unpack the operators
    Sx,Sy,Sz = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz
    
    #-----------------#
    D:Symmetry = shift(SpinOp)
    print(repr(D))
    nblocks, _ = D.count_blocks()
    print("\tnblocks:",nblocks)
    D.diagonalize(method="dense") # 'dense' i smuch better than 'jacobi'
    # D.eigenvalues.sort()
    # print(D.eigenvalues)
    l,N = D.energy_levels()
    # print(len(l))
    assert len(l) == NSpin, "wrong number of energy levels"
    ru = np.sort(roots_of_unity(len(spin_values)))
    l  = np.sort(l)    
    assert np.allclose(l,ru), "The eigenvalues should be the roots of the unity."
    

    #-----------------#
    # Heisenberg Hamiltonian along the z-axis
    H = Heisenberg(Sx,Sy,Sz) 
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
    E,Psi = H.diagonalize_with_symmetry(S=[D])
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

if __name__ == "__main__":
    test_Ising_symmetries()
