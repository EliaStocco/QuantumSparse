import pytest
import numpy as np
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin.shift import shift
from quantumsparse.conftest import *
from quantumsparse.cli.test.generate_hermitean_operators import hermitian_basis

@parametrize_N
@parametrize_S
@parametrize_interaction
def test_simple(S, N, interaction):

    # spin operators
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)
    H: Operator = get_H(Sx, Sy, Sz, interaction=interaction)

    H.diagonalize()

    assert np.allclose(H.trace(), np.sum(H.eigenvalues)), (
        f"Trace mismatch after diagonalization: "
        f"Tr(H)={H.trace()}, ΣE={np.sum(H.eigenvalues)}"
    )

    assert np.allclose((H @ H).trace(), np.sum(H.eigenvalues**2)), (
        f"Second moment mismatch after diagonalization: "
        f"Tr(H²)={(H @ H).trace()}, ΣE²={np.sum(H.eigenvalues**2)}"
    )

    D: Symmetry = shift(SpinOp)
    H.diagonalize_with_symmetry(D)

    assert np.allclose(H.trace(), np.sum(H.eigenvalues)), (
        f"Trace mismatch after symmetry diagonalization: "
        f"Tr(H)={H.trace()}, ΣE={np.sum(H.eigenvalues)}"
    )

    assert np.allclose((H @ H).trace(), np.sum(H.eigenvalues**2)), (
        f"Second moment mismatch after symmetry diagonalization: "
        f"Tr(H²)={(H @ H).trace()}, ΣE²={np.sum(H.eigenvalues**2)}"
    )
    
    T = np.geomspace(0.1, 1000, 10)

    C = H.thermal_average(T, H @ H) - H.thermal_average(T, H)**2
    
    tol = 1e-12
    negative = C < -tol

    assert np.all(~negative), (
        f"Heat capacity variance should be non-negative (tol={tol:g}).\n"
        + "\n".join(
            f"T={t:.6g}, C={c:.6e}"
            for t, c in zip(T[negative], C[negative])
        )
    )
    
    average = H.thermal_average(T, H)
    C2 = np.zeros_like(T)
    for n, t in enumerate(T):
        delta = H - average[n] * H.iden()
        C2[n] = H.thermal_average(np.asarray([t]), delta @ delta)
        
    assert np.allclose(C,C2)
        
    
@parametrize_N
@parametrize_S
@parametrize_interaction
def test_complex(S, N, interaction):

    # spin operators
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)
    H: Operator = get_H(Sx, Sy, Sz, interaction=interaction)
    
    basis:List[Operator] = hermitian_basis(H.shape[0],10)
    H.diagonalize()
    T = np.geomspace(0.1, 1000, 10)
    
    for Op in basis:
        v = H.thermal_average(np.asarray([np.inf]),Op)[0]
        assert np.allclose(Op.trace()/Op.shape[0], v)
        
        C = H.thermal_average(T, Op @ Op) - H.thermal_average(T, Op)**2

        tol = 1e-12
        negative = C < -tol

        assert np.all(~negative), (
            f"Heat capacity variance should be non-negative (tol={tol:g}).\n"
            + "\n".join(
                f"T={t:.6g}, C={c:.6e}"
                for t, c in zip(T[negative], C[negative])
            )
        )
    


if __name__ == "__main__":
    pytest.main([__file__])