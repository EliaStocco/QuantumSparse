import pytest
import json
from quantumsparse.matrix import Matrix
from quantumsparse.tools.bookkeeping import TOLERANCE as TOLERANCE
from pathlib import Path

test_dir = Path(__file__).parent.parent/"tests"


parametrize_method      = pytest.mark.parametrize("method", ["R", "U"])
parametrize_interaction = pytest.mark.parametrize("interaction",  [
                             "heisenberg", "DM", "biquadratic", "anisotropy", #"rhombicity",
                             "heisenberg DM", 
                             "heisenberg biquadratic", 
                             "heisenberg DM biquadratic",
                             # test_dir/"Cr8-U.json"
                         ])
parametrize_N           = pytest.mark.parametrize("N", [2, 3, 4])
parametrize_S           = pytest.mark.parametrize("S", [0.5, 1.0, 1.5, 2.0])

COUPLINGS = [1.0,2.1,3.5] # just some numbers

def check_diagonal(H:Matrix):
    norm = H.test_eigensolution().norm()
    assert norm < TOLERANCE, f"Diagonalization failed: norm = {norm}"

from typing import List, Tuple
import numpy as np
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.spin import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg, anisotropy, rhombicity

def get_H(Sx:List[Operator],Sy:List[Operator],Sz:List[Operator],interaction:str)->Operator:
    
    H = 0 
    
    np.random.seed(42)
    def random_couplings()->List[float]:
        return np.random.rand(3).tolist()
    
   # Convert to Path safely
    interaction_path = Path(interaction)

    # -------- FILE CASE --------
    if interaction_path.is_file():
        
        with open(interaction_path, "r") as f:
            interactions:dict = json.load(f)
            
            if "factor" in interactions:
                factor = interactions.pop("factor")
                print(f"Using conversion factor: {factor}.")
            
            for key,value in interactions.items():
                if isinstance(value, list):
                    interactions[key] = factor * np.asarray(value)
                else:
                    interactions[key] = factor * value
    
            print("Creating Hamiltonian operator:")
            H = 0 # Sx[0].empty()
            if "heisenberg" in interactions:
                print("Adding Heisenberg interaction (1nn) with couplings ", interactions["heisenberg"])
                H += Heisenberg(Sx,Sy,Sz,couplings=interactions["heisenberg"],nn=1)
            if "heisenberg-2" in interactions and len(Sx) > 2:
                print("Adding Heisenberg interaction (2nn) with couplings ", interactions["heisenberg-2"])
                H += Heisenberg(Sx,Sy,Sz,couplings=interactions["heisenberg-2"],nn=2)
            if "heisenberg-3" in interactions and len(Sx) > 3:
                print("Adding Heisenberg interaction (3nn) with couplings ", interactions["heisenberg-3"])
                H += Heisenberg(Sx,Sy,Sz,couplings=interactions["heisenberg-3"],nn=3)
            if "heisenberg-4" in interactions and len(Sx) > 4:
                print("Adding Heisenberg interaction (4nn) with couplings ", interactions["heisenberg-4"])
                H += Heisenberg(Sx,Sy,Sz,couplings=interactions["heisenberg-4"],nn=4)
            if "DM" in interactions:
                print("Adding Dzyaloshinskii-Moriya interaction with couplings ", interactions["DM"])
                H += Dzyaloshinskii_Moriya(Sx,Sy,Sz,couplings=interactions["DM"])
            if "biquadratic" in interactions:
                print("Adding biquadratic interaction with couplings ", interactions["biquadratic"])
                H += biquadratic_Heisenberg(Sx,Sy,Sz,couplings=interactions["biquadratic"])
            if "zeeman" in interactions:
                
                print("Computing the magnetic moments ... ", end="")
                Mx, My, Mz = magnetic_moments(Sx, Sy, Sz)
                print("done.")
                
                print("Adding Zeeman interaction with couplings ", interactions["zeeman"])
                B = interactions["zeeman"]
                H += -(Mx*B[0] + My*B[1] + Mz*B[2])
                
            if not isinstance(H, Operator):
                raise ValueError(f"The Hamiltonian operator was not constructed properly. No interactions were added ({interaction}).")
            
            return H
        
    # -------- STRING CASE --------
    interaction = str(interaction)
            
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
        
    if not isinstance(H, Operator):
        raise ValueError(f"The Hamiltonian operator was not constructed properly. No interactions were added ({interaction}).")
        
    return H

def NS2Ops(N:int,S:float)->Tuple[List[Operator],List[Operator],List[Operator],SpinOperators]:
    spin_values = np.full(N, S)
    SpinOp = SpinOperators(spin_values)
    Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz
    return Sx,Sy,Sz, SpinOp

def template_test_script(script_name:str):
    import subprocess
    result = subprocess.run(
        [script_name, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Command should exit successfully
    assert result.returncode == 0

    # Help text should contain usage
    assert "usage" in result.stdout.lower()
    