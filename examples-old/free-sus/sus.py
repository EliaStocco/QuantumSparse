import numpy as np
import matplotlib.pyplot as plt
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.spin.interactions import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg, Ising
from quantumsparse.statistics import susceptibility, Curie_constant

S     = 0.5
Nsites = 4
spin_values = np.full(Nsites, S)
SpinOp = SpinOperators(spin_values)

spins = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
Sx, Sy, Sz = spins

Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)

SpinOp.basis

# H = Sx[0].empty()
H = Heisenberg(Sx,Sy,Sz,1)
H.diagonalize()
H.eigenvalues = H.eigenvalues.real
H = H.sort()
H

temp = np.linspace(1,300,100)
sus = susceptibility(temp, H, Mz)

C = Curie_constant(spin_values)

fig,ax = plt.subplots(figsize=(4,4))
# ax.plot(temp,sus*temp,label=r"$\chi$T")
ax.plot(temp,sus,label=r"$\chi$",color="blue")
ax.hlines(C,0,300,label="Curie",color="red")
plt.grid()
plt.legend()
plt.show()
pass
