import argparse
import json
import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.spin.interactions import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg

def main():
    
    description = "Diagonalize an operator."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-j", "--interactions"   , type=str, required=True , help="JSON file with the interactions.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="pickle output file with the operator.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")  
    
    print(f"Reading interactions from file '{args.interactions}' ... ", end="")
    with open(args.interactions, "r") as f:
        interactions:dict = json.load(f)
    print("done.")
            
    if "factor" in interactions:
        factor = interactions.pop("factor")
        print(f"Using conversion factor: {factor}.")
    
    for key,value in interactions.items():
        if isinstance(value, list):
            interactions[key] = factor * np.asarray(value)
        else:
            interactions[key] = factor * value
    
    print("Creating Hamiltonian operator ... ", end="")
    H = 0
    if "heisenberg" in interactions:
        print("Adding Heisenberg interaction (1nn) with couplings ", interactions["heisenberg"])
        H += Heisenberg(*SpinOp.spins,couplings=interactions["heisenberg"],nn=1)
    if "heisenberg-2" in interactions:
        print("Adding Heisenberg interaction (2nn) with couplings ", interactions["heisenberg-2"])
        H += Heisenberg(*SpinOp.spins,couplings=interactions["heisenberg-2"],nn=2)
    if "heisenberg-3" in interactions:
        print("Adding Heisenberg interaction (3nn) with couplings ", interactions["heisenberg-3"])
        H += Heisenberg(*SpinOp.spins,couplings=interactions["heisenberg-3"],nn=3)
    if "heisenberg-4" in interactions:
        print("Adding Heisenberg interaction (4nn) with couplings ", interactions["heisenberg-4"])
        H += Heisenberg(*SpinOp.spins,couplings=interactions["heisenberg-4"],nn=4)
    if "DM" in interactions:
        print("Adding Dzyaloshinskii-Moriya interaction with couplings ", interactions["DM"])
        H += Dzyaloshinskii_Moriya(*SpinOp.spins,couplings=interactions["DM"])
    if "biquadratic" in interactions:
        print("Adding biquadratic interaction with couplings ", interactions["biquadratic"])
        H += biquadratic_Heisenberg(*SpinOp.spins,couplings=interactions["biquadratic"])
    if "zeeman" in interactions:
        
        print("Computing the magnetic moments ... ", end="")
        Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)
        print("done.")
        
        print("Adding Zeeman interaction with couplings ", interactions["zeeman"])
        B = interactions["zeeman"]
        H += -(Mx*B[0] + My*B[1] + Mz*B[2])
    
    print("done.")    
    assert isinstance(H, Operator), "The Hamiltonian operator was not constructed properly."
    
    print(f"Saving operator to file '{args.output}' ... ", end="")
    H.save(args.output)
    print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("construct_hamiltonian")