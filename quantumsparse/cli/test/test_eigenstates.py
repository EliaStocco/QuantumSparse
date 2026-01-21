import argparse
import os
import pandas as pd
import numpy as np
from quantumsparse.operator import Operator

def main():
    
    description = "Test eigenstates"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information (with the 'hermitean-operators' folder too).")
    parser.add_argument("-io", "--input_operator"   , type=str, required=True , help="pickle input file with the operator.")
    parser.add_argument("-T", "--temperatures"   , type=float, nargs='+', required=False , help="list of temperatures (default: %(default)s).", default=[1.,10.,100.,1000.])
    parser.add_argument("-o", "--output"      , type=str, required=True, help="csv output file with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    folder = args.input_spins + "/hermitean-operators"
    Hilbert_space_dimension = H.shape[0]**2
    results = []
    T = np.asarray(args.temperatures)
    for i in range(Hilbert_space_dimension):
        file = f"{folder}/op_{i}.pickle"
        if not os.path.exists(file):
            break
            
        print(f"Computing thermal average of operator {i} ... ", end="")
        O = Operator.load(file)
        res = H.thermal_average(operator=O,temperatures=T).real
        results.append(res)
        print("done.")

    results = np.array(results)
    df = pd.DataFrame(results,columns=[f"T={T}" for T in args.temperatures])
    print(f"Saving results to file '{args.output}' ... ", end="")
    df.to_csv(args.output,index=False)
    print("done.")
        
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("test_eigenstates")