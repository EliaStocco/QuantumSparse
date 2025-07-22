# %%
import numpy as np
import os
import logging
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Symmetry, Operator
from quantumsparse.spin.shift import shift
from quantumsparse.spin.interactions import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg
from quantumsparse.spin.functions import rotate_spins, get_unitary_rotation_matrix
from quantumsparse.tools.mathematics import product, roots_of_unity

DEBUG = False
TOLERANCE = 1e-10

# Configure logging
logging.basicConfig(
    filename="LOGFILE",               # <-- Log to this file
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'                        # Optional: overwrite on each run
)

logger = logging.getLogger()  # Root logger

# %% [markdown]
# ## Parameters

# %%
name = "NAME"
xc = "LDA+U"
interaction = INTERACTIONS
oname = "OUTPUT"
ofolder = "FOLDER"
os.makedirs(ofolder,exist_ok=True)

all_Js = {
    "Cr8" : {
        "LDA+U"   : [0.596,0.596,0.843],
        "LDA+U+V" : [0.848,0.848,1.198],
    },
    "V8"  : {
        "LDA+U"   : [-0.643,-0.654,-0.913],
        "LDA+U+V" : [-0.403,-0.403,-0.585],
    }
}

all_bi = {
    "Cr8" : {
        "LDA+U"   : [-0.009,0.005,0.026],
        "LDA+U+V" : None
    },
    "V8"  : {
        "LDA+U"   : [-0.23,-0.208,0.091],
        "LDA+U+V" : None
    }
}

all_DM = {
    "Cr8" : {
        "LDA+U"   : [-0.193,-0.193,0.608],
        "LDA+U+V" : [-0.282,-0.282,0.853],
    },
    "V8"  : {
        "LDA+U"   : [-0.060,-0.050,-1.049],
        "LDA+U+V" : [-0.175,-0.170,-0.813],
    }
}

S     = 3./2 if name == "Cr8" else 1
Nsites = 8
spin_values = np.full(Nsites, S)

def build_H(Sx, Sy, Sz):
    logger.debug("Building Hamiltonian.")
    Js = all_Js[name][xc]
    BI = all_bi[name][xc]
    DM = all_DM[name][xc]
    logger.debug("Adding Heisenberg interactions.")
    h = Heisenberg(Sx, Sy, Sz, Js)
    if "biquad" in interaction:
        logger.debug("Adding biquadratic interactions.")
        h = h + biquadratic_Heisenberg(Sx, Sy, Sz, BI)
    if "dm" in interaction:
        logger.debug("Adding Dzyaloshinskii-Moriya interactions.")
        h = h + Dzyaloshinskii_Moriya(Sx, Sy, Sz, DM)
    return h.clean()

# %% [markdown]
# ## Spin operators

# %%
logger.info("Creating spin operators.")
SpinOp = SpinOperators(spin_values)
logger.debug("Spin operators created with spin values: %s", spin_values)

spins = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
Sx, Sy, Sz = spins

SpinOp.basis

# %%
assert isinstance(Sx[0], Operator), "Sx[0] should be an Operator instance"
Sx[0]

# %% [markdown]
# ## Hamiltonian

# %%
logger.info("Constructing Hamiltonian (Cartesian frame).")
H = build_H(Sx, Sy, Sz)
logger.debug("Hamiltonian constructed.")
H

# %% [markdown]
# ## Translational symmetry

# %%
logger.info("Building or loading shift (symmetry) operator.")
filename = f"D.S={S}.N={Nsites}.pickle"
if os.path.exists(filename):
    logger.debug("Loading shift operator from file.")
    D = Symmetry.load(filename)
else:
    logger.debug("Creating shift operator.")
    D: Symmetry = shift(SpinOp)
D

# %%
if not D.is_diagonalized():
    logger.info("Diagonalizing shift operator.")
    D.diagonalize(method="dense")

if DEBUG:
    logger.debug("Testing symmetry operator diagonalization.")
    test = D.test_eigensolution()
    assert test.norm() < TOLERANCE, f"Symmetry operator D is not diagonalized correctly: {test.norm()}"

# %%
if DEBUG:
    logger.debug("Validating energy levels.")
    test = D.test_eigensolution()
    norm = test.norm()
    assert norm < TOLERANCE, f"Symmetry operator D is not diagonalized correctly: {norm}"

    l, N = D.energy_levels()
    assert len(l) == Nsites, "Wrong number of energy levels"

    ru = np.sort(roots_of_unity(len(spin_values)))
    l = np.sort(l)
    assert np.allclose(l, ru), "Eigenvalues should be roots of unity"

# %%
if not os.path.exists(filename):
    logger.info("Saving symmetry operator to file.")
    D.save(filename)

# %% [markdown]
# ## Diagonalizing the Hamiltonian

# %%
if DEBUG:
    logger.debug("Testing Hamiltonian and symmetry commutator.")
    comm = Operator.commutator(H, D)
    assert comm.norm() < TOLERANCE, "Commutator is not zero"

if not H.is_diagonalized():
    logger.info("Diagonalizing Hamiltonian with symmetry.")
    H.diagonalize_with_symmetry(S=[D], method="dense")

if DEBUG:
    logger.debug("Testing Hamiltonian eigensolution.")
    test = H.test_eigensolution()
    norm = test.norm()
    assert norm < TOLERANCE, f"Hamiltonian not diagonalized correctly: {norm}"

H

# %%
logger.info("Saving Cartesian Hamiltonian.")
H.save(f"{ofolder}/H.{oname}.cart.pickle")

# %% [markdown]
# ## Cylindrical coordinates

# %%
logger.info("Applying cylindrical rotation to spin operators.")
EulerAngles = np.zeros((Nsites, 3))
EulerAngles[:, 2] = np.linspace(0, 360, Nsites, endpoint=False)
EulerAngles = np.pi * EulerAngles / 180
logger.debug("Euler angles (radians):\n%s", EulerAngles)

if not os.path.exists(f"U/{name}/Utot.pickle"):
    U, Ud = get_unitary_rotation_matrix(spins, EulerAngles)

    Utot = product(U).clean()
    logger.debug("Global unitary operator constructed.")

    os.makedirs("U/", exist_ok=True)
    os.makedirs(f"U/{name}", exist_ok=True)
    for n, u in enumerate(U):
        u.save(f"U/{name}/U.n={n}.pickle")
    Utot.save(f"U/{name}/Utot.pickle")
    logger.info("Unitary operators saved.")
else:
    logger.debug("Loading unitary operator from file.")
    Utot = Operator.load(f"U/{name}/Utot.pickle")

# %%
if DEBUG:
    logger.debug("Testing rotations using U and R methods.")
    StR, SrR, SzR = rotate_spins(spins, EulerAngles=EulerAngles, method="R")
    StU, SrU, SzU = rotate_spins(spins, EulerAngles=EulerAngles, method="U")
    for n in range(Nsites):
        assert (StR[n] - StU[n]).norm() < TOLERANCE, "St rotation mismatch"
        assert (SrR[n] - SrU[n]).norm() < TOLERANCE, "Sr rotation mismatch"
        assert (SzR[n] - SzU[n]).norm() < TOLERANCE, "Sz rotation mismatch"

    UdTot = product(Ud).clean()
    assert (Utot - UdTot.dagger()).norm() < TOLERANCE, "Utot and UdTot mismatch"

    for n in range(Nsites):
        assert (StR[n] - Utot @ Sx[n] @ UdTot).norm() < TOLERANCE
        assert (SrR[n] - Utot @ Sy[n] @ UdTot).norm() < TOLERANCE
        assert (SzR[n] - Utot @ Sz[n] @ UdTot).norm() < TOLERANCE

# %%
if DEBUG:
    logger.info("Building Hamiltonian in cylindrical coordinates.")
    Hcyl = build_H(StR, SrR, SzR)
    Hcyl

# %%
if DEBUG:
    logger.info("Comparing block count.")
    print("n of blocks of H   : ", H.count_blocks()[0])
    print("n of blocks of Hcyl: ", Hcyl.count_blocks()[0])

# %%
if DEBUG:
    Htest = H.unitary_transformation(Utot)
    test = Hcyl - Htest
    norm = test.norm()
    assert norm < TOLERANCE, f"Cylindrical Hamiltonian is incorrect: {norm}"

# %%
logger.info("Final unitary transformation to cylindrical frame.")
Hfinal = H.unitary_transformation(Utot)
if DEBUG:
    test = Hfinal.test_eigensolution()
    assert test.norm() < TOLERANCE, f"Hamiltonian eigensolution error: {test.norm()}"

Hfinal

# %%
logger.info("Saving cylindrical Hamiltonian.")
Hfinal.save(f"{ofolder}/H.{oname}.cyl.pickle")

# %% [markdown]
# ## Density of states
