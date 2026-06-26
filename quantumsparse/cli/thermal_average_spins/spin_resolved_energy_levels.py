import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from quantumsparse.tools import resolved_energy_levels

def main():
    
    description = "Compute the spin resolved energy levels."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="folder filled by 'prepare_thermal_average_spins.py'.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output folder with the spin resolved energy levels.")
    args = parser.parse_args()
        
    print(f"\n=== {description} ===\n")
    
    files = [file for file in os.listdir(args.input)]
    Nsites = int(len(files) / 3)
    
    os.makedirs(args.output, exist_ok=True)    
        
    for site in range(Nsites):
        for n, xyz in enumerate(["x", "y", "z"]):
        
            pfile = f"{args.input}/S{xyz}_{site}.csv"
            print(f"Reading results from file '{pfile}' ... ", end="")
            prepare_df = pd.read_csv(pfile)
            print("done.")
            
            rows = resolved_energy_levels(
                prepare_df[["eigenvalues","A"]].to_numpy(),
                {
                    "eigenvalues": 1e-8,
                    "A": 1e-4,
                },
            )

            df = pd.DataFrame(rows)
            ofile = f"{args.output}/S{xyz}_{site}.csv"
            print(f"Saving results to file '{ofile}' ... ", end="")
            df.to_csv(ofile,index=False)
            print("done.")
                 
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()

def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("spin_resolved_energy_levels")