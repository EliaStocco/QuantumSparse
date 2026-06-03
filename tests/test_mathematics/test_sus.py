import pytest
import numpy as np
import matplotlib.pyplot as plt

from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin.shift import shift
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.statistics import T2beta
from quantumsparse.constants import kB
from quantumsparse.tools.bookkeeping import TOLERANCE
from quantumsparse.conftest import *


@parametrize_N
@parametrize_S
@parametrize_interaction
def test_sus(N, S, interaction):

    # --- Build spin operators ---
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)

    H0: Operator = get_H(Sx, Sy, Sz, interaction=interaction)

    # symmetry (used consistently everywhere)
    D: Symmetry = shift(SpinOp)

    H0.diagonalize_with_symmetry(D)
    H0 = H0.clean()
    check_diagonal(H0)
    assert H0.test_eigensolution().norm() < TOLERANCE

    # --- magnetization operators ---
    # Mx, My, Mz = (
    #     sum(SpinOp.Sx),
    #     sum(SpinOp.Sy),
    #     sum(SpinOp.Sz),
    # )
    Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)

    directions = ["x", "y", "z"]
    Ms = [Mx, My, Mz]
    # directions = ["z"]
    # Ms = [Mz]

    Tmin, Tmax = 1, 1000
    temperatures = np.logspace(np.log10(Tmin), np.log10(Tmax), 100)

    eps = 1e-4  # small field for finite difference

    for direction, M in zip(directions, Ms):

        # -----------------------------
        # 1) Numerical derivative at b=0
        # -----------------------------
        H_plus:Operator = H0.clone() - eps * M
        H_minus:Operator = H0.clone() + eps * M

        H_plus.diagonalize_with_symmetry(D)
        H_minus.diagonalize_with_symmetry(D)
        
        H_plus = H_plus.clean()
        H_minus = H_minus.clean()
        

        A_plus = H_plus.thermal_average(temperatures, M)
        A_minus = H_minus.thermal_average(temperatures, M)
        
        assert H_plus.test_eigensolution().norm() < 1e-6
        assert H_minus.test_eigensolution().norm() < 1e-6
        
        # if (H_minus.eigenstates - H_plus.eigenstates).norm() == 0:
        #     pytest.skip()

        derivative = (A_plus - A_minus) / (2 * eps) * kB * temperatures

        # -----------------------------
        # 2) FDT at b = 0 ensemble
        # -----------------------------
        A  = H0.thermal_average(temperatures, M)
        AB = H0.thermal_average(temperatures, M @ M.dagger())

        beta = T2beta(temperatures)
        fdt =  (AB - A**2)


        # # -----------------------------
        # # 2) FDT at b = 0 ensemble
        # # -----------------------------
        # A  = H0.thermal_average(temperatures, M)
        

        # beta = T2beta(temperatures)

        # fdt = np.zeros_like(temperatures)

        # for n, t in enumerate(temperatures):

        #     # centered operator at this temperature
        #     M_centered = M - A[n] * M.iden()

        #     # variance via operator square
        #     fdt[n] = H0.thermal_average(np.asarray([t]), M_centered @ M_centered)[0]


        # -----------------------------
        # 3) Comparison
        # -----------------------------
        # diff = np.abs(derivative - fdt)
        # ii = diff > 1e-10
        # diff = diff[ii] #np.abs(diff[ii] / fdt[ii]) # relative
        if not np.allclose(derivative, fdt, atol=1e-6):
            
            diff = derivative - fdt

            msg = (
                f"\nFDT mismatch (direction={direction})\n"
                f"max|diff| = {np.max(np.abs(diff)):.3e}\n"
                f"mean|diff| = {np.mean(np.abs(diff)):.3e}"
            )

            # # optional debug plot
            plt.plot(temperatures, derivative, label="derivative")
            plt.plot(temperatures, fdt, label="FDT")
            plt.xscale("log")
            plt.yscale("log")
            plt.title(f"N={N}, S={S}, dir={direction}")
            plt.legend()
            plt.show()

            raise AssertionError(msg)

        # proceed to next direction

if __name__ == "__main__":
    pytest.main([__file__])
