import argparse
import pandas as pd
import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.statistics import statistical_weights

def main():
    
    description = "Save the statistical weights to file."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-ih", "--input_hamiltonian", type=str, required=True , help="pickle file with the Hamiltonian.")
    parser.add_argument("-t", "--temperatures_file"   , type=str, required=True , help="txt file with the temperatures in Kelvin.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="csv output file with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading Hamiltonian operator from file '{args.input_hamiltonian}' ... ", end="")
    H = Operator.load(args.input_hamiltonian)
    H = H.sort()
    print("done.")
    
    print(f"Reading temperatures from file '{args.temperatures_file}' ... ", end="")
    temp = np.loadtxt(args.temperatures_file)
    print("done.")
    assert temp.ndim == 1, f"'{args.temperatures_file}' should contain a 1D array."
    print("n. temperatures: ",len(temp))
    
    w, Z = statistical_weights(T=temp,E=H.eigenvalues)
    
    results = pd.DataFrame(data=w.T,index=H.eigenvalues, columns=temp )
    
    print(f"Saving results to file '{args.output}' ... ",end="")
    results.to_csv(args.output)
    print("done")
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_thermal_average")