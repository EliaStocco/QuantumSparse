import argparse
import numpy as np
import pandas as pd
from quantumsparse.tools import energy_levels

def main():
    
    description = "Find the energy levels from the eigenvalues."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-e", "--eigenvalues"   , type=str, required=True , help="txt input file with the eigenvalues.")
    parser.add_argument("-t", "--tolerance", type=float, required=False, help="energy tolerance (default: %(default)s).", default=1e-8)
    parser.add_argument("-o", "--output"   , type=str, required=True , help="txt output file.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading data from file '{args.eigenvalues}' ... ",end="")
    eigenvalues = np.loadtxt(args.eigenvalues,dtype=complex)
    print("done")
    assert np.allclose(eigenvalues.imag,0), "The imaginary part is not zero."
    eigenvalues = eigenvalues.real
    
    print("Computing energy levels ... ", end="")
    l,n = energy_levels(eigenvalues,args.tolerance)
    print("done")
    
    df = pd.DataFrame({"degeneracy":n,"energy":l})
    print(f"Saving energy levels to file '{args.output}' ... ",end="")
    df.to_csv(args.output,index=False)
    print("done")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("energy_levels")