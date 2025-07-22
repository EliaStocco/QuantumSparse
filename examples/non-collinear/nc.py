# %% [markdown]
# # Non collinear calculations

# %% [markdown]
# ## Packages

# %%
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Symmetry, Operator
from quantumsparse.spin.shift import shift
from quantumsparse.spin.interactions import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg
from quantumsparse.spin.functions import rotate_spins, get_unitary_rotation_matrix
from quantumsparse.tools.mathematics import product, roots_of_unity
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('../../notebook.mplstyle')                # use matplotlib style notebook.mplstyle

DEBUG = False
TOLERANCE = 1e-10

# %% [markdown]
# ## Parameters

# %% [markdown]
# Let's define the main parameters of the system: the spin value `S` and the number of sites `Nsites`.

# %%
name = "NAME"
xc = "LDA+U"
interaction = INTERACTIONS
oname = "OUTPUT"
all_Js = {
    "Cr8" : {
        "LDA+U"   : [0.596,0.596,0.843], # meV
        "LDA+U+V" : [0.848,0.848,1.198], # eV
    },
    "V8"  : {
        "LDA+U"   : [-0.643,-0.654,-0.913], # meV
        "LDA+U+V" : [-0.403,-0.403,-0.585], # meV
    }
}

all_bi = {
    "Cr8" : {
        "LDA+U"   : [-0.009,0.005,0.026], # meV
        "LDA+U+V" : None
        
    },
    "V8"  : {
        "LDA+U"   : [-0.23,-0.208,0.091], # meV
        "LDA+U+V" : None
    }
}

all_DM = {
    "Cr8" : {
        "LDA+U"   : [-0.193,-0.193,0.608], # meV
        "LDA+U+V" : [-0.282,-0.282,0.853], # meV
        
    },
    "V8"  : {
        "LDA+U"   : [-0.060,-0.050,-1.049], # meV
        "LDA+U+V" : [-0.175,-0.170,-0.813], # meV
    }
}

S     = 3./2 # spin value
Nsites = 4 # number of sites
spin_values = np.full(Nsites,S)

def build_H(Sx,Sy,Sz):
    Js = all_Js[name][xc]
    BI = all_bi[name][xc]
    DM = all_DM[name][xc]
    h =     Heisenberg(Sx,Sy,Sz,Js)
    if "biquad" in interaction:
        h = h + biquadratic_Heisenberg(Sx,Sy,Sz,BI)
    if "dm" in interaction:
        h = h + Dzyaloshinskii_Moriya(Sx,Sy,Sz,DM)
    return h.clean()

# %% [markdown]
# ## Spin operators

# %% [markdown]
# From these values we can construct the spin operators `Sx`, `Sy`, and `Sz` of the system (in cartesian coordinates).

# %%
# construct the spin operators
SpinOp = SpinOperators(spin_values)

# unpack the operators
spins = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz
Sx,Sy,Sz = spins

# %%
# let's show the Hilbert space basis:
# each row is a basis state
# each column if one component in the Sz-basis
SpinOp.basis

# %% [markdown]
# Pay attention that `Sx` (as well as `Sy`, and `Sz`) are `numpy.array` with lenght `Nsites`, and each one of its element is a `Operator` object.
# 
# Let's inspect one element.

# %%
assert isinstance(Sx[0],Operator), "Sx[0] should be an Operator instance"
Sx[0]

# %% [markdown]
# ## Hamiltonian

# %% [markdown]
# Let's construct the Hamiltonian.

# %%
# cylindricar coordinates
H = build_H(Sx,Sy,Sz)
H

# %% [markdown]
# ## Translational symmetry

# %% [markdown]
# Let's construct the shift operator (or traslation operator) because it will be usefull later on to make the diagonalization of the Hamiltoninan cheaper.

# %%
if os.path.exists(f"D.S={S}.N={Nsites}.pickle"):
    D = Symmetry.load(f"D.S={S}.N={Nsites}.pickle") # load the symmetry operator from a file
else:
    D:Symmetry = shift(SpinOp)
D

# %% [markdown]
# Let's diagonalize the shift operator so that we have access to its eigenvectors.

# %%
if not D.is_diagonalized():
    D.diagonalize(method="dense") # 'dense' is much better than 'jacobi'
    
if DEBUG:
    test = D.test_eigensolution()
    assert test.norm() < TOLERANCE, f"Symmetry operator D is not diagonalized correctly: {test.norm()}"

D

# %%
if DEBUG :
    
    test = D.test_eigensolution()
    norm  = test.norm()
    assert norm < TOLERANCE, f"Symmetry operator D is not diagonalized correctly: {norm}"

    # the number of energy levels should be equal to the number of sites
    l,N = D.energy_levels()
    # print(len(l))
    assert len(l) == Nsites, "wrong number of energy levels"

    # the eigenvalues should be the roots of unity
    ru = np.sort(roots_of_unity(len(spin_values)))
    l  = np.sort(l)    
    assert np.allclose(l,ru), "The eigenvalues should be the roots of the unity."


# test

# %% [markdown]
# Let' save the shift operator, and its eigensolutions to file.

# %%
if not os.path.exists(f"D.S={S}.N={Nsites}.pickle"):
    D.save(f"D.S={S}.N={Nsites}.pickle") # save the symmetry operator to a file

# %% [markdown]
# ## Diagonalizing the Hamiltonian

# %%
if DEBUG:
    comm = Operator.commutator(H,D)
    assert comm.norm() < TOLERANCE, "Commutator is not zero, the symmetry operator does not commute with the Hamiltonian"

if not H.is_diagonalized(): 
    H.diagonalize_with_symmetry(S=[D],method="dense"); # diagonalize the Hamiltonian
    
if DEBUG:
    test = H.test_eigensolution()
    norm = test.norm()
    assert norm < TOLERANCE, f"Hamiltonian is not diagonalized correctly: {norm}"
    
H

# %%
H.save(f"H.{oname}.cart.pickle") # save the Hamiltonian to a file

# %% [markdown]
# ## Cylindrical coordinates

# %%
EulerAngles = np.zeros((Nsites,3))
EulerAngles[:,2] = np.linspace(0,360,Nsites,endpoint=False)
EulerAngles = np.pi * EulerAngles / 180
print("Euler angles (in radians):\n", EulerAngles)

# %%
StR,SrR,SzR= rotate_spins(spins,EulerAngles=EulerAngles,method="R")
U, Ud = get_unitary_rotation_matrix(spins, EulerAngles)
Utot = product(U).clean() # this is really memory intensive
Utot

os.makedirs("U/")
os.makedirs(f"U/{oname}")
for n,u in enumerate(U):
    u.save(f"U/{oname}/U.n={n}.pickle") # save to file
Utot.save(f"U/{oname}/Utot.pickle")
    

# %%
if DEBUG:
    # StR,SrR,SzR= rotate_spins(spins,EulerAngles=EulerAngles,method="R")
    StU,SrU,SzU= rotate_spins(spins,EulerAngles=EulerAngles,method="U")
    for n in range(Nsites):
        print(f"Site {n}:")
        assert (StR[n] - StU[n]).norm() < TOLERANCE, "St rotation mismatch"
        assert (SrR[n] - SrU[n]).norm() < TOLERANCE, "Sr rotation mismatch"
        assert (SzR[n] - SzU[n]).norm() < TOLERANCE, "Sz rotation mismatch"

    # Utot = product(U).clean()
    UdTot = product(Ud).clean()
    assert (Utot - UdTot.dagger()).norm() < TOLERANCE, "Utot and UdTot mismatch"

    for n in range(Nsites):
        print(f"Site {n}:") 
        assert (StR[n] - Utot @ Sx[n] @ UdTot).norm() < TOLERANCE, "St rotation mismatch"
        assert (SrR[n] - Utot @ Sy[n] @ UdTot).norm() < TOLERANCE, "Sr rotation mismatch"
        assert (SzR[n] - Utot @ Sz[n] @ UdTot).norm() < TOLERANCE, "Sz rotation mismatch"

# %%
Hcyl = build_H(StR,SrR,SzR)
Hcyl

# %%
print("n of blocks of H   : ", H.count_blocks()[0])
print("n of blocks of Hcyl: ", Hcyl.count_blocks()[0])

# %%
if DEBUG:
    Htest = H.unitary_transformation(Utot)
    test = Hcyl - Htest
    norm = test.norm()
    assert norm < TOLERANCE, f"Hamiltonian in cylindrical coordinates is not correct: {norm}"
    Htest

# %%
Hfinal = H.unitary_transformation(Utot) # from cartesian to cylindrical frame
if DEBUG:
    test = Hfinal.test_eigensolution()
    norm = test.norm()
    assert norm < TOLERANCE, f"Hamiltonian in cylindrical coordinates is not correct: {norm}"
Hfinal

# %%

Hfinal.save(f"H.{oname}.cyl.pickle") # save the Hamiltonian to a file

# %% [markdown]
# ## Density of states
