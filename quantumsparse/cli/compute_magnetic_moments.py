import argparse
import os
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments

def main():
    
    description = "Construct the magnetic moments operators."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="output folder with the Mx, My and Mz  operators.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")

    # symmetry operator (shift)
    print("Computing the magnetic moments ... ", end="")
    Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)
    print("done.")
    
    print(f"Saving magnetic moments operators to folder '{args.output}' ... ", end="")
    os.makedirs(args.output,exist_ok=True)
    Mx.save(f"{args.output}/Mx.pickle")
    My.save(f"{args.output}/My.pickle")
    Mz.save(f"{args.output}/Mz.pickle")
    print("done.")
    
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_magnetic_moments")