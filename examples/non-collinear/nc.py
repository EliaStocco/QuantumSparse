import logging
import numpy as np
import os
import matplotlib.pyplot as plt

from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Symmetry, roots_of_unity, Operator
from quantumsparse.spin.shift import shift
from quantumsparse.spin import Heisenberg

plt.style.use('../../notebook.mplstyle')

logging.basicConfig(
    filename='nc.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # <-- now includes year, month, day
)

def gaussian_dos(l, N=None, sigma=0.1, num_points=1000):
    logging.info(f"Computing Gaussian DOS with sigma={sigma}, num_points={num_points}")
    l = np.asarray(l)
    if N is None:
        logging.info("No degeneracies provided, using uniform weights")
        N = np.ones_like(l)
    else:
        logging.info(f"Using provided degeneracies, total weight sum={N.sum()}")
    E_min, E_max = l.min() - 2*sigma, l.max() + 2*sigma
    logging.info(f"Energy range for DOS: [{E_min:.3f}, {E_max:.3f}]")
    E_vals = np.linspace(E_min, E_max, num_points)
    dos = np.zeros_like(E_vals)
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    for i, (energy, weight) in enumerate(zip(l, N), start=1):
        dos += weight * prefactor * np.exp(-0.5 * ((E_vals - energy) / sigma)**2)
        if i % max(1, len(l)//10) == 0 or i == len(l):
            logging.debug(f"Processed Gaussian for energy level {i}/{len(l)} (energy={energy:.3f})")
    return E_vals, dos

all_Js = {
    "Cr8": {
        "LDA+U": [0.596, 0.596, 0.843],
        "LDA+U+V": [0.848, 0.848, 1.198],
    },
    "V8": {
        "LDA+U": [-0.643, -0.654, -0.913],
        "LDA+U+V": [-0.403, -0.403, -0.585],
    }
}

all_S = {
    "Cr8": 3. / 2.,
    "V8": 1,
}

for name in ["Cr8"]:
    logging.info(f"Starting processing for system '{name}'")

    S = all_S[name]
    Nsites = 8
    spin_values = np.full(Nsites, S)
    logging.info(f"Spin value S={S}, Number of sites={Nsites}")

    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
    logging.info("Spin operators created")

    assert isinstance(Sx[0], Operator), "Sx[0] should be an Operator instance"

    D_file = f"D.S={S}.N={Nsites}.pickle"
    if os.path.exists(D_file):
        logging.info(f"Loading symmetry operator from {D_file}")
        D = Symmetry.load(D_file)
    else:
        logging.info(f"Computing symmetry operator for S={S}, N={Nsites}")
        D = shift(SpinOp)
        logging.info("Diagonalizing symmetry operator...")
        D.diagonalize(method="dense")
        test = D.test_eigensolution()
        norm = test.norm()
        logging.info(f"Symmetry operator eigen test norm: {norm:.3e}")
        assert norm < 1e-10, "Symmetry operator eigen test failed"

        l, N = D.energy_levels()
        logging.info(f"Symmetry energy levels count: {len(l)}")
        assert len(l) == Nsites, "Wrong number of energy levels"

        ru = np.sort(roots_of_unity(len(spin_values)))
        l = np.sort(l)
        assert np.allclose(l, ru), "Eigenvalues should be roots of unity"

        D.save(D_file)
        logging.info(f"Symmetry operator saved to {D_file}")

    for xc in ["LDA+U+V"]:
        logging.info(f"Starting Hamiltonian calculation for functional '{xc}'")
        H_file = f"H.{name}.xc={xc}.pickle"

        if os.path.exists(H_file):
            logging.info(f"Loading Hamiltonian from {H_file}")
            H = Operator.load(H_file)
        else:
            Js = all_Js[name][xc]
            logging.info(f"Building Heisenberg Hamiltonian with Js={Js}")
            H = Heisenberg(Sx, Sy, Sz, Js)

            comm = Operator.commutator(H, D)
            comm_norm = comm.norm()
            logging.info(f"Commutator norm between H and D: {comm_norm:.3e}")
            assert comm_norm < 1e-10, "Symmetry operator does not commute with Hamiltonian"

            logging.info("Diagonalizing Hamiltonian with symmetry")
            H.diagonalize_with_symmetry(S=[D], method="dense")

            test = H.test_eigensolution()
            norm = test.norm()
            logging.info(f"Hamiltonian eigen test norm: {norm:.3e}")
            assert norm < 1e-10, "Hamiltonian eigen test failed"

            H.save(H_file)
            logging.info(f"Hamiltonian saved to {H_file}")

        l, N = H.energy_levels()
        weight_sum = N.sum()
        expected_size = H.shape[0]
        logging.info(f"Energy levels found: {len(l)}, sum of weights: {weight_sum}, expected Hilbert space size: {expected_size}")
        assert weight_sum == expected_size, "Sum of energy level weights mismatch"

        E_vals, dos = gaussian_dos(l, N, sigma=0.1)

        plt.plot(E_vals, dos, label=f"{name} {xc}")
        plt.xlabel('energy [eV]')
        plt.ylabel('DOS')
        plt.title(f'{name} - {xc} DOS (σ=0.1 eV)')
        plt.grid(True)
        pdf_file = f"{name}.xc={xc}.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        logging.info(f"DOS plot saved to {pdf_file}")
        # plt.show()
        plt.clf()  # Clear the current figure for the next plot
