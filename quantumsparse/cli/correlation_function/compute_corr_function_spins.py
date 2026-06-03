import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from quantumsparse.statistics import dfT2correlation_function

def main():
    
    description = "Compute the correlation function between spin operators (assumes translational invariance)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", type=str, required=True, help="folder with the results produced by 'prepare_corr_function_spins.py'.")
    parser.add_argument("-t", "--temperatures_file", type=str, required=True, help="txt file with the temperatures in Kelvin.")
    parser.add_argument("-o", "--output", type=str, required=True, help="output folder with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")

    # ----------------------------
    # Load temperatures
    # ----------------------------
    print(f"Reading temperatures from '{args.temperatures_file}' ... ", end="")
    temp = np.loadtxt(args.temperatures_file)
    print("done.")
    assert temp.ndim == 1, "temperatures file should contain a 1D array"
    print(f"→ n. temperatures: {len(temp)}")

    # ----------------------------
    # Load instructions
    # ----------------------------
    index_file = f"{args.input}/index.csv"
    print(f"\nReading instructions from '{index_file}' ... ", end="")
    instructions = pd.read_csv(index_file)
    print("done.")
    print(f"→ n. operator pairs: {len(instructions)}")

    # ----------------------------
    # Compute correlations
    # ----------------------------
    print("\nComputing thermal correlations ...")

    results_by_pair = {}

    for idx, row in instructions.iterrows():

        A = row["component_A"]
        B = row["component_B"]
        r = row["distance"]
        fname = row["file"]

        print(f"  [{idx+1}/{len(instructions)}] processing ({A},{B}, r={r}) ... ", end="")

        df = pd.read_csv(os.path.join(args.input, fname))
        C_T = dfT2correlation_function(temp, df)

        key = (A, B)
        if key not in results_by_pair:
            results_by_pair[key] = {}

        results_by_pair[key][r] = C_T

        print("done.")

    # ----------------------------
    # Save results
    # ----------------------------
    print(f"\nWriting output to '{args.output}' ...")
    os.makedirs(args.output, exist_ok=True)

    for (A, B), data in results_by_pair.items():

        print(f"  saving pair ({A},{B}) ... ", end="")

        distances = sorted(data.keys())

        table = {"T": temp}
        for r in distances:
            table[f"r={r}"] = data[r]

        df_out = pd.DataFrame(table)

        filename = os.path.join(args.output, f"C_{A}{B}.csv")
        df_out.to_csv(filename, index=False)

        print(f"done → {os.path.basename(filename)}")

    print("\nJob done :)\n")


if __name__ == "__main__":
    main()


def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_corr_function_spins")