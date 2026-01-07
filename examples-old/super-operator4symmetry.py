import numpy as np
from quantumsparse.spin import SpinOperators, from_S2_to_S, spin2dim
from quantumsparse.operator import Operator
from quantumsparse.operator import Symmetry, roots_of_unity
from quantumsparse.spin import Heisenberg, anisotropy, DM
from quantumsparse.spin.shift import shift
from quantumsparse.hilbert import HilbertSpace, LocalHilbertSpace, embed_operators
import pytest

@pytest.mark.parametrize("S,NSpin", [(2,4),(3,3),(5,2)])
def main(S=0.5,NSpin=3):
    
    spin_values = np.full(NSpin,S)
    dims = spin2dim(spin_values)
    
    minimal = np.zeros((NSpin,NSpin))
    for i in range(NSpin):
        j = i+1 if i < NSpin-1 else 0
        minimal[i,j] = 1
    w,f = np.linalg.eig(minimal)
    
    SpinOps = SpinOperators(spin_values)
    Sz = SpinOps.Sz
    D = shift(SpinOps)
    W,F = D.diagonalize()
    
    #-----------------------------#
    HS = HilbertSpace(dims)
    OpBasis = HS.get_operator_basis()
    
    TEST = D.empty()
    M = len(OpBasis[0])
    for i in range(NSpin):
        for j in range(NSpin):
            TEST += minimal[i,j] * Sz[j] #* Sz[i].dagger()
            # j = i+1 if i < NSpin-1 else 0
            # for m in range(M):
            #     TEST += minimal[i,j] * OpBasis[j][m] * OpBasis[i][m]#.dagger()
                
    # Proj = OpBasis[0][0].dagger() @ OpBasis[0][0]
    # for n in range(1,len(OpBasis[0])):
    #     Proj += OpBasis[0][n].dagger() @ OpBasis[0][n]
        
    Projs = HS.get_projectors()
        
    # #-----------------------------#    
    # LHS = [ LocalHilbertSpace(d) for d in dims ]     
    # LocOpBasis = [ lhs.get_operator_basis() for lhs in LHS ]
    
    # proj = LocOpBasis[0][0].dagger() @ LocOpBasis[0][0]
    # for n in range(1,len(LocOpBasis[0])):
    #     proj += LocOpBasis[0][n].dagger() @ LocOpBasis[0][n]
        
    # test  = embed_operators([[proj]],dims)
    
    # norm = (test[0][0] - Proj).norm() # this should NOT be zero
    
    #-----------------------------#

    # construct the spin operators
    SpinOp = SpinOperators(spin_values)
    # unpack the operators
    Sx,Sy,Sz = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz
    
    #-----------------#
    D:Symmetry = shift(SpinOp)
    print(repr(D))
    nblocks, _ = D.count_blocks()
    print("\tnblocks:",nblocks)
    D.diagonalize() # 'dense' is smuch better than 'jacobi'
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
    
    H = H.sort()
    Hnosym = Hnosym.sort()
    
    assert np.allclose(H.eigenvalues,Hnosym.eigenvalues), "The eigenvalues should be the same."
    assert np.allclose(H.eigenstates,Hnosym.eigenstates), "The eigenstates should be the same."
    
    return
    

if __name__ == "__main__":
    
    main()

