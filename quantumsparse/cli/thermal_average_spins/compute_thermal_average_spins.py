import argparse
import os
import pandas as pd
import numpy as np
from quantumsparse.statistics import dfT2thermal_average_and_fluctuation

def main():
    
    description = "Compute the thermal average of all spin operators."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input"   , type=str, required=True , help="folder filles by 'prepare_thermal_average_spins.py'.")
    parser.add_argument("-t", "--temperatures_file"   , type=str, required=True , help="txt file with the temperatures in Kelvin.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="output folder with the results.")
    args = parser.parse_args()
    
    assert args.input != args.output, f"Input and output folders can not be the same."
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading temperatures from file '{args.temperatures_file}' ... ", end="")
    temp = np.loadtxt(args.temperatures_file)
    print("done.")
    assert temp.ndim == 1, f"'{args.temperatures_file}' should contain a 1D array."
    print("n. temperatures: ",len(temp))
    
    os.makedirs(args.output,exist_ok=True)
    
    for file in os.listdir(args.input):
        
        ifile = f"{args.input}/{file}"
        print(f"Reading results from file '{ifile}' ... ", end="")
        df = pd.read_csv(ifile)
        print("done.")
        # print("n. eigenvalues: ",len(df))
        
        ave, fluc = dfT2thermal_average_and_fluctuation(temp,df)
        results = pd.DataFrame({"temp":temp,"average":ave,"fluctuation":fluc})
        
        ofile = f"{args.output}/{file}"
        print(f"Saving results to file '{ofile}' ... ",end="")
        results.to_csv(ofile,index=False)
        print("done\n")
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_thermal_average")