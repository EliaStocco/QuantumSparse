import numpy as np
from quantumsparse.operator import Operator,  Symmetry
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.statistics import susceptibility

hfile = "H.V8.ex.cyl.pickle"
H = Operator.load(hfile)

sfile = "D.S=1.N=8.pickle"
S = Symmetry.load(sfile)

Eks, Hks = H.band_diagram(S)

pass