import numpy as np
from quantumsparse.spin import SpinOperators, from_S2_to_S
from quantumsparse.operator import Operator
from quantumsparse.operator import Symmetry, roots_of_unity
from quantumsparse.spin import Heisenberg, anisotropy, DM
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
def test_Heisenberg_symmetries(S=1,NSpin=4):
    
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
    
    H = Heisenberg(Sx,Sy,Sz,couplings=[1,2,3]) + anisotropy(Sz) + DM(Sx,Sy,Sz,couplings=[12,23,35])
    print(repr(H))
    
    nblocks, _ = H.count_blocks()
    print("\tnblocks:",nblocks)    
    
    #-----------------#
    # save an indipendent copy of the Hamiltonian 
    # to check that nothing fishy in going on
    # (numpy arrays can be insidious)
    Hnosym = Operator(H.copy())
    
    assert np.all(H.data == Hnosym.data), "the data should be the same"
    assert H is not Hnosym, "the variables should be independent"
    
    E,Psi = H.diagonalize_with_symmetry(S=[D])
    Enosym,Psinosym = Hnosym.diagonalize()
    
    H.sort(inplace=True)
    Hnosym.sort(inplace=True)
    
    assert np.allclose(H.eigenvalues,Hnosym.eigenvalues), "The eigenvalues should be the same."
    assert np.allclose(H.eigenstates,Hnosym.eigenstates), "The eigenstates should be the same."
    
    return
    

if __name__ == "__main__":
    
    test_Heisenberg_symmetries()

