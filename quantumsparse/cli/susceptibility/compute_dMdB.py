import argparse
import os
import pandas as pd
import numpy as np

def main():
    
    description = "Compute the magnetic susceptibility as -dM/dB."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--inputs"   , type=str, required=True , help="csv input files produced by 'compute_thermal_average.py'.")
    parser.add_argument("-B", "--Bfields"  , type=str, required=True , help="B fields [T]")
    parser.add_argument("-d", "--pol_deg"  , type=int, required=False, help="degree of the polynomial fit used to compute the derivative (default: %(default)s).", default=2)
    parser.add_argument("-o", "--output"   , type=str, required=False, help="output folder with the results (default: %(default)s).", default="dMdB")
    args = parser.parse_args()
    
    args.inputs = [ str(x) for  x in str(args.inputs).split(" ")]
    args.Bfields = [ float(x) for  x in str(args.Bfields).split(" ")]
    
    if 0.00 not in args.Bfields:
        raise ValueError("B=0.00 T must be included in the list of B fields.")
    
    print(f"\n=== {description} ===\n")
    
    all_df = None
    fluc_df = None

    for file, B in zip(args.inputs, args.Bfields):
        print(f"\nReading results from file '{file}' ... ", end="")
        df = pd.read_csv(file)
        print("done.")
        print("n. temperatures:", len(df))

        # Initialize dataframes on first iteration
        if all_df is None:
            all_df = pd.DataFrame()
            fluc_df = pd.DataFrame()

            all_df["temp"] = df["temp"]
            fluc_df["temp"] = df["temp"]

        # Optional safety check
        if not np.allclose(all_df["temp"], df["temp"]):
            raise ValueError(f"Temperature mismatch in file {file}")

        # Store averages
        all_df[B] = df["average"]

        # Store fluctuations
        fluc_df[B] = df["fluctuation"]
    
    os.makedirs(args.output, exist_ok=True)
    print(f"Saving averages to file '{args.output}/averages.csv' ... ", end="")
    all_df.to_csv(f"{args.output}/averages.csv", index=False)
    print("done.")
    
    print(f"Saving fluctuations to file '{args.output}/fluctuations.csv' ... ", end="")
    fluc_df.to_csv(f"{args.output}/fluctuations.csv", index=False)
    del fluc_df
    print("done.")
    
    # -------------------------------
    # Compute dM/dB at B = 0
    # -------------------------------

    print("\nComputing dM/dB at B = 0 ...")

    B_vals = np.array(sorted(args.Bfields))
    B_columns = B_vals  # column names are floats

    # Ensure polynomial degree is valid
    if args.pol_deg >= len(B_vals):
        raise ValueError(
            f"Polynomial degree ({args.pol_deg}) must be < number of B fields ({len(B_vals)})."
        )

    # Extract B values (columns) and M matrix
    B_vals = np.array(sorted(args.Bfields))
    B_columns = B_vals  # column names are floats

    # M matrix: rows = temperatures, columns = B
    M = all_df[B_columns].to_numpy(dtype=float)  # shape (nT, nB)

    # Construct Vandermonde matrix for polynomial fitting
    # Each row of V: [B^deg, B^(deg-1), ..., B^0]
    V = np.vander(B_vals, N=args.pol_deg+1, increasing=False)  # shape (nB, deg+1)

    # Solve least-squares for all temperatures at once
    # We want coefficients: a0*B^n + a1*B^(n-1) + ... + an
    # Using np.linalg.lstsq in a loop-free fashion
    coeffs_all = np.linalg.lstsq(V, M.T, rcond=None)[0]  # shape (deg+1, nT)

    # Derivative coefficients: multiply by power for each term
    powers = np.arange(args.pol_deg, 0, -1)[:, None]  # shape (deg, 1)
    dcoeffs_at_B = coeffs_all[:-1, :] * powers        # remove constant term

    # Evaluate derivative at B=0
    # Only the coefficient of B^1 contributes at B=0, which is the linear term
    # So take the last row of dcoeffs_at_B corresponding to B^1
    # If deg=2: coeffs_all = [a2, a1, a0], derivative = [2*a2, a1], dM/dB @0 = a1
    chi = dcoeffs_at_B[-1, :]  # shape (nT,)

    chi_df = pd.DataFrame({
        "temp": all_df["temp"],
        "X": -chi
    })

    # Compute chi x T
    chi_df["XT"] = chi_df["X"] * chi_df["temp"]
    
    print(f"Saving susceptibility to file '{args.output}/chi.csv' ... ", end="")
    chi_df.to_csv(f"{args.output}/chi.csv", index=False)
    print("done.")

    
    # ---------------------------------------
    # Save fitted magnetization M^fit(T,B)
    # ---------------------------------------

    print("Computing fitted magnetization M^fit(T,B) ...")

    Mfit_df = pd.DataFrame()
    Mfit_df["temp"] = all_df["temp"]

    # Prepare empty columns for each B
    for B in B_vals:
        Mfit_df[B] = np.nan

    # Loop over temperatures
    for i, row in all_df.iterrows():
        M_vals = row[B_columns].values.astype(float)

        # Fit polynomial
        coeffs = np.polyfit(B_vals, M_vals, deg=args.pol_deg)

        # Evaluate fitted polynomial at all B fields
        M_fit_vals = np.polyval(coeffs, B_vals)

        # Store fitted values
        Mfit_df.loc[i, B_columns] = M_fit_vals

    print(f"Saving fitted magnetization to file '{args.output}/M_fit.csv' ... ", end="")
    Mfit_df.to_csv(f"{args.output}/M_fit.csv", index=False)
    print("done.")
    
        # -------------------------------
    # Compute M/B
    # -------------------------------
    del all_df[0]
    print("\nComputing M/B ... ", end="")
    for B in args.Bfields:
        if B == 0.0:
            continue
        all_df[B] = all_df[B] / B
    print("done.")
    
    print(f"Saving M/B to file '{args.output}/M_over_B.csv' ... ", end="")
    all_df.to_csv(f"{args.output}/M_over_B.csv", index=False)
    print("done.")
    
    # -------------------------------
    # Compute T M/B
    # -------------------------------
    print("\nComputing TM/B ... ", end="")
    for B in args.Bfields:
        if B == 0.0:
            continue
        all_df[B] = all_df["temp"] * all_df[B]
    print("done.")
    
    print(f"Saving TM/B to file '{args.output}/TM_over_B.csv' ... ", end="")
    all_df.to_csv(f"{args.output}/TM_over_B.csv", index=False)
    print("done.")

    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("compute_dMdB")