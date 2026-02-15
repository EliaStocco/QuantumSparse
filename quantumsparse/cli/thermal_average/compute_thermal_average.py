import argparse
import pandas as pd
import numpy as np
from quantumsparse.statistics import dfT2thermal_average_and_fluctuation

def main():
    
    description = "Compute the thermal average of an operator."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input"   , type=str, required=True , help="csv input file produced by 'prepare_thermal_average.py'.")
    parser.add_argument("-t", "--temperatures_file"   , type=str, required=True , help="txt file with the temperatures in Kelvin.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="csv output file with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading results from file '{args.input}' ... ", end="")
    df = pd.read_csv(args.input)
    print("done.")
    print("n. eigenvalues: ",len(df))
    
    print(f"Reading temperatures from file '{args.temperatures_file}' ... ", end="")
    temp = np.loadtxt(args.temperatures_file)
    print("done.")
    assert temp.ndim == 1, f"'{args.temperatures_file}' should contain a 1D array."
    print("n. temperatures: ",len(temp))
    
    ave, fluc = dfT2thermal_average_and_fluctuation(temp,df)
    
    results = pd.DataFrame({"temp":temp,"average":ave,"fluctuation":fluc})
    
    print(f"Saving results to file '{args.output}' ... ",end="")
    results.to_csv(args.output,index=False)
    print("done")
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_thermal_average")