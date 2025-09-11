import numpy as np
import pytest
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.spin import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg, anisotropy, rhombicity
from quantumsparse.tools.debug import compare_eigensolutions_dense_real

@pytest.mark.parametrize("interaction", ["heisenberg", "DM","biquadratic","anisotropy","rhombicity"])
@pytest.mark.parametrize("N, S", [
    (2, 0.5),
    (3, 0.5),
    (4, 0.5),
    (4, 1.0),
])
def test_heisenberg_hamiltonian_with_vs_without_blocks(N: int, S: float,interaction:str) -> Operator:
    """
    Build a Hamiltonian for a ring of N spins of spin-S.
    """
    spin_values = np.full(N, S)
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
    
    def get_H(Sx,Sy,Sz):
        if interaction == "heisenberg":
            H = Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=[1,2,3])
        elif interaction == "DM":
            H = Dzyaloshinskii_Moriya(Sx=Sx, Sy=Sy, Sz=Sz,couplings=[1,2,3])
        elif interaction == "biquadratic":
            H = biquadratic_Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=[1,2,3])
        elif interaction == "anisotropy":
            H = anisotropy(Sz=Sz)
        elif interaction == "rhombicity":
            H = rhombicity(Sx=Sx, Sy=Sy)
        else:
            raise ValueError(f"Unknown interaction: {interaction}")
        return H

    H = get_H(Sx=Sx, Sy=Sy, Sz=Sz)
    eigenvalues, eigenvectors = H.diagonalize()
    
    Hdense = H.todense()
    eigenvalues_dense, eigenvectors_dense = np.linalg.eigh(Hdense)

    # test
    compare_eigensolutions_dense_real(eigenvalues,eigenvalues_dense,
                                 eigenvectors.todense(),eigenvectors_dense,
                                 Hdense)
    
if __name__ == "__main__":
    pytest.main([__file__])