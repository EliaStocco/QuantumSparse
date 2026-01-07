import argparse
import json
import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.spin import SpinOperators
from quantumsparse.spin import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg, anisotropy, rhombicity

def main():
    
    description = "Diagonalize an operator."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input"         , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-j", "--interactions"   , type=str, required=True , help="JSON file with the interactions.")
    parser.add_argument("-o", "--output"      , type=str, required=False, help="pickle output file (default: %(default)s).", default=None)
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading spins from folder '{args.input}' ... ", end="")
    SpinOp = SpinOperators.load(args.input)
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
        H += Heisenberg(*SpinOp.spins,couplings=interactions["heisenberg"])
    if "DM" in interactions:
        H += Dzyaloshinskii_Moriya(*SpinOp.spins,couplings=interactions["DM"])
    if "biquadratic" in interactions:
        H += biquadratic_Heisenberg(*SpinOp.spins,couplings=interactions["biquadratic"])
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