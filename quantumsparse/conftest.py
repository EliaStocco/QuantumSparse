import pytest

parametrize_method      = pytest.mark.parametrize("method", ["R", "U"])
parametrize_interaction = pytest.mark.parametrize("interaction",  [
                             "heisenberg", "DM", "biquadratic", "anisotropy", "rhombicity",
                             "heisenberg DM", 
                             "heisenberg biquadratic", 
                             "heisenberg DM biquadratic",
                         ])
parametrize_N           = pytest.mark.parametrize("N", [2, 3, 4])
parametrize_S           = pytest.mark.parametrize("S", [0.5, 1.0, 1.5, 2.0])

TOLERANCE = 1e-10
COUPLINGS = [1.0,2.1,3.5] # just some numbers

from typing import List, Tuple
import numpy as np
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.spin import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg, anisotropy, rhombicity

def random_couplings()->List[float]:
    return np.random.rand(3)

def get_H(Sx:List[Operator],Sy:List[Operator],Sz:List[Operator],interaction:str)->Operator:
    H = 0 
    if "heisenberg" in interaction:
        H += Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=random_couplings())
    if "DM" in interaction:
        H += Dzyaloshinskii_Moriya(Sx=Sx, Sy=Sy, Sz=Sz,couplings=random_couplings())
    if "biquadratic" in interaction:
        H += biquadratic_Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=random_couplings())
    if "anisotropy" in interaction:
        H += anisotropy(Sz=Sz)
    if "rhombicity" in interaction:
        H += rhombicity(Sx=Sx, Sy=Sy)
    return H

def NS2Ops(N:int,S:float)->Tuple[List[Operator],List[Operator],List[Operator],SpinOperators]:
    spin_values = np.full(N, S)
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
    return Sx,Sy,Sz, SpinOp