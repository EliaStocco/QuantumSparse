import argparse
import json
import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.spin.interactions import Heisenberg, Dzyaloshinskii_Moriya, biquadratic_Heisenberg

def main():
    
    description = "Add Zeeman interaction to a Hamiltonian."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str , required=True , help="folder with the spin information.")
    parser.add_argument("-io", "--input_operator", type=str , required=True , help="pickle input file with the operator.")
    parser.add_argument("-B", "--Bfield"         ,type=float, nargs=3,required=True,metavar=("Bx", "By", "Bz"),help="Magnetic field components in Tesla (Bx By Bz)")
    parser.add_argument("-o", "--output"         , type=str , required=True, help="pickle output file with the operator.")
    args = parser.parse_args()
     
    if len(args.Bfield) != 3:
        parser.error("Bfield requires exactly 3 components: Bx By Bz")
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")  
    
    print(f"Reading operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    print("Computing the magnetic moments ... ", end="")
    Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)
    print("done.")
    
    Bx, By, Bz = args.Bfield
    print(f"Adding Zeeman interaction with couplings {Bx} {By} {Bz}")
    H += -(Mx*Bx + My*By + Mz*Bz)
    
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
    template_test_script("add_Zeeman")