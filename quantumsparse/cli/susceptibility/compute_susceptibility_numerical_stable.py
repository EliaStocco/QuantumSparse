import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from quantumsparse.statistics import corr2sus
from quantumsparse.operator import Operator

def main():
    
    description = "Compute the magnetic susceptibility in a numerical stable way."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-ih", "--input_hamiltonian", type=str, required=True , help="pickle file with the Hamiltonian.")
    parser.add_argument("-ia", "--input_a"   , type=str, required=True , help="pickle file with the first operator.")
    parser.add_argument("-ib", "--input_b"   , type=str, required=False , help="pickle file with the second operator (default: same as 'input_a').", default=None)
    parser.add_argument("-t", "--temperatures_file"   , type=str, required=True , help="txt file with the temperatures in Kelvin.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="csv output file with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading Hamiltonian operator from file '{args.input_hamiltonian}' ... ", end="")
    H = Operator.load(args.input_hamiltonian)
    print("done.")
    
    print(f"Reading operator A from file '{args.input_a}' ... ", end="")
    Ma = Operator.load(args.input_a)
    print("done.")
    
    if args.input_b is not None:
        print(f"Reading operator B from file '{args.input_b}' ... ", end="")
        Mb = Operator.load(args.input_b)
        print("done.")
    else:
        Mb = None
        
    print(f"Reading temperatures from file '{args.temperatures_file}' ... ", end="")
    temp = np.loadtxt(args.temperatures_file)
    print("done.")
    assert temp.ndim == 1, f"'{args.temperatures_file}' should contain a 1D array."
    print("n. temperatures: ",len(temp))   
    
    
    meanA = H.thermal_average(temp,Ma)
    if Mb is not None:
        meanB = H.thermal_average(temp,Mb)
    else:
        meanB = meanA
        
    N = len(H)
    Id = Operator.identity(N)
    results = pd.DataFrame({"temp":temp,"corr":np.zeros_like(temp)})
    for n,T in tqdm(enumerate(temp),desc="Computing susceptibility",total=len(temp)):
        
        deltaA = Ma - meanA[n]* Id   
        if Mb is not None:
            deltaB = Mb - meanB[n] * Id
        else:
            deltaB = deltaA
            
        product = deltaA @ deltaB
        
        corr = H.thermal_average(T,product)
        results.at[n,"corr"] = float(corr)
    
    sus = corr2sus(temp,results["corr"].to_numpy())
    results["X"] = sus
    results["XT"] = temp*sus
    
    print(f"Saving results to file '{args.output}' ... ",end="")
    results.to_csv(args.output,index=False)
    print("done")
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_susceptibility_numerical_stable")