from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin.shift import shift
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import rotate_spins, get_unitary_rotation_matrix
from quantumsparse.tools.mathematics import product, roots_of_unity
from quantumsparse.tools.quantum_mechanics import expectation_value
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.statistics import susceptibility
import numpy as np
import matplotlib.pyplot as plt

Nsites = 4

SpinOp = SpinOperators(np.full(Nsites,1))
spins = SpinOp.Sx,SpinOp.Sy,SpinOp.Sz
Sx,Sy,Sz = spins

EulerAngles = np.zeros((Nsites,3))
EulerAngles[:,2] = np.linspace(0,360,Nsites,endpoint=False)
EulerAngles = np.pi * EulerAngles / 180
print("Euler angles (in radians):\n", EulerAngles)

U, Ud = get_unitary_rotation_matrix(spins, EulerAngles)
Utot = product(U).clean() # this is really memory intensive


# H = Operator.load("H.V8.ex.cyl.pickle")
# H.clean()
D = Symmetry.load("D.S=1.N=4.pickle")
U = Symmetry.load("Utot.pickle")
H = Operator.load("H.V4.ex.cyl.pickle") / 1000

k, Hk= H.band_diagram(D)

Ek = [hk.eigenvalues for hk in Hk ]

# # plt.style.use('../../notebook.mplstyle')
# fig,ax = plt.subplots(figsize=(4,3))
# for n,E in enumerate(Ek):
#     ax.scatter(np.full(len(E),n),E)
# ax.grid()
# plt.show()

Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)
temp = np.geomspace(0.0001, 300.,100)
sus = susceptibility(temp, H, Mz,Mz)

sz = expectation_value(Sz.sum(),H.eigenstates)

H.clean()
repr(H)
repr(H.eigenstates)

# test = H.test_eigensolution()

D = shift(SpinOp)
D.diagonalize()
H.band_diagram(D)
pass