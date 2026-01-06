import pytest

parametrize_method      = pytest.mark.parametrize("method", ["R", "U"])
parametrize_interaction = pytest.mark.parametrize("interaction", ["heisenberg", "DM", "biquadratic", "anisotropy", "rhombicity"])
parametrize_N           = pytest.mark.parametrize("N", [2, 3, 4])
parametrize_S           = pytest.mark.parametrize("S", [0.5, 1.0, 1.5, 2.0])

TOLERANCE = 1e-10
COUPLINGS = [1.0,2.1,3.5] # just some numbers

from typing import List
from quantumsparse.operator import Operator
from quantumsparse.spin import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg, anisotropy, rhombicity

def get_H(Sx:List[Operator],Sy:List[Operator],Sz:List[Operator],interaction:str)->Operator:
    H = 0 
    if interaction == "heisenberg":
        H += Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=COUPLINGS)
    elif interaction == "DM":
        H += Dzyaloshinskii_Moriya(Sx=Sx, Sy=Sy, Sz=Sz,couplings=COUPLINGS)
    elif interaction == "biquadratic":
        H += biquadratic_Heisenberg(Sx=Sx, Sy=Sy, Sz=Sz,couplings=COUPLINGS)
    elif interaction == "anisotropy":
        H += anisotropy(Sz=Sz)
    elif interaction == "rhombicity":
        H += rhombicity(Sx=Sx, Sy=Sy)
    else:
        raise ValueError(f"Unknown interaction: {interaction}")
    return H