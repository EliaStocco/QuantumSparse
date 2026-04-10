import argparse
import pandas as pd
import numpy as np
from quantumsparse.operator import Operator

def main():
    
    description = "Compute the expectation value of a operator on each eigenstate of an Hamiltonian."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-ih", "--input_hamiltonian", type=str, required=True , help="pickle file with the Hamiltonian.")
    parser.add_argument("-io", "--input_operator"   , type=str, required=True , help="pickle file with the operator.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="csv output file with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading Hamiltonian from file '{args.input_hamiltonian}' ... ", end="")
    H = Operator.load(args.input_hamiltonian)
    H = H.sort()
    print("done.")
    
    print(f"Reading operator from file '{args.input_operator}' ... ", end="")
    Op = Operator.load(args.input_operator)
    print("done.")
    
    print("Computing expectation values ... ",end="")
    values = H.expectation_value(Op)
    print("done.")
    
    df = pd.DataFrame({
        "energy" : H.eigenvalues.real,
        "values.real" : values.real,
        "values.imag" : values.imag,
        })
    
    print(f"Saving results to file '{args.output}' ... ",end="")
    df.to_csv(args.output,index=False)
    print("done.")
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_thermal_average")