
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Symmetry, Operator
from quantumsparse.spin.shift import shift, shift_foundamental
from quantumsparse.spin import Heisenberg, Ising
from quantumsparse.spin.functions import rotate_spins, get_unitary_rotation_matrix
from quantumsparse.tools.mathematics import product
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

plt.style.use('../../notebook.mplstyle')                # use matplotlib style notebook.mplstyle


S     = 1./2. # spin value
Nsites = 4 # number of sites
spin_values = np.full(Nsites,S)


T = shift_foundamental(Nsites)
T


SpinOps = SpinOperators(spin_values)
spins = SpinOps.Sx, SpinOps.Sy, SpinOps.Sz
Sx = SpinOps.Sx
Sy = SpinOps.Sy
Sz = SpinOps.Sz


EulerAngles = np.zeros((8,3))
EulerAngles[:,2] = 360 - np.linspace(0,360,8,endpoint=False)
EulerAngles = np.pi * EulerAngles / 180
print("Euler angles (in radians):\n", EulerAngles)

print("Delta Euler angles (in deg):\n", np.diff(EulerAngles,axis=0)*180/np.pi)

U, Ud = get_unitary_rotation_matrix(spins, EulerAngles)
StR,SrR,SzR= rotate_spins(spins,EulerAngles=EulerAngles,method="R")
StU,SrU,SzU= rotate_spins(spins,EulerAngles=EulerAngles,method="U")

tol = 1e-12  # or whatever precision you consider acceptable
for n in range(Nsites):
    assert (StR[n] - StU[n]).norm() < tol, "St rotation mismatch"
    assert (SrR[n] - SrU[n]).norm() < tol, "Sr rotation mismatch"
    assert (SzR[n] - SzU[n]).norm() < tol, "Sz rotation mismatch"

Utot = product(U)
UdTot = product(Ud)
assert (Utot - UdTot.dagger()).norm() < tol, "Utot and UdTot mismatch"

for n in range(Nsites):
    assert (StR[n] - Utot @ Sx[n] @ UdTot).norm() < tol, "St rotation mismatch"
    assert (SrR[n] - Utot @ Sy[n] @ UdTot).norm() < tol, "Sr rotation mismatch"
    assert (SzR[n] - Utot @ Sz[n] @ UdTot).norm() < tol, "Sz rotation mismatch"
