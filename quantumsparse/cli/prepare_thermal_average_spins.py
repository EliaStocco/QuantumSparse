import argparse
import os
import numpy as np
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.cli.prepare_thermal_average import compute_preparation

def main():
    
    description = "Prepare the calculation of the thermal average of all spin operators."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-io", "--input_operator", type=str, required=True , help="pickle input file with the operator.")
    parser.add_argument("-o", "--output"         , type=str, required=True, help="output folder with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading Hamiltonian operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    assert H.is_diagonalized(), "Hamiltonian should be diagonalized."
    assert H.is_hermitean(), "The operator is not hermitean"
    assert np.allclose(H.eigenvalues.imag,0.), "The eigenvalues of the Hamiltonian should be real."
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")
    
    N = SpinOp.nsites
    os.makedirs(args.output,exist_ok=True)
    for n,(sx,sy,sz) in enumerate(zip(SpinOp.Sx,SpinOp.Sy,SpinOp.Sz)):
        print(f"Processing spins for site {n+1}/{N} ... ",end="")
        for op,name in zip([sx,sy,sz],["x","y","z"]):
            file = f"{args.output}/S{name}_{n}.csv"
            df = compute_preparation(H,op,show=False)
            df.to_csv(file,index=False)
        print("done.")
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("prepare_thermal_average_spins")