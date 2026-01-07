#!/usr/bin/env python3
import argparse
import numpy as np
import logging
import pandas as pd
from quantumsparse.operator import Operator
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments
from quantumsparse.statistics import susceptibility

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Compute magnetic susceptibility.")
    parser.add_argument("-i", "--input", required=True, help="Input file (pickle format).")
    parser.add_argument("-o", "--output", required=True, help="Output file for susceptibility data (.npz).")
    parser.add_argument("-N", type=int, required=True, help="Number of spin sites.")
    parser.add_argument("-S", type=float, required=True, help="Spin value per site.")
    parser.add_argument("--Tmin", type=float, default=1e-8, help="Minimum temperature (default: 1e-8).")
    parser.add_argument("--Tmax", type=float, default=300.0, help="Maximum temperature (default: 300).")
    parser.add_argument("--Tpoints", type=int, default=10, help="Number of temperature points (default: 10).")
    parser.add_argument(
        "--operators", nargs=2, choices=["Mx", "My", "Mz"], default=["Mx", "Mx"],
        help="Two operators for susceptibility calculation (e.g., Mx Mz). Default: Mx Mx."
    )
    parser.add_argument("--logfile", default="susceptibility.log", help="Log file name (default: susceptibility.log).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        filename=args.logfile,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger()

    logger.info("Starting susceptibility calculation.")
    logger.info(f"Input Hamiltonian: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Spin value: S = {args.S}, Number of sites: N = {args.N}")
    logger.info(f"Temperature range: {args.Tmin} to {args.Tmax}, Points: {args.Tpoints}")
    logger.info(f"Operators used: {args.operators[0]}, {args.operators[1]}")
    logger.info(f"Logging to: {args.logfile}")
    if args.debug:
        logger.debug("Debug mode is ON.")

    # Load the Hamiltonian
    logger.info("Loading Hamiltonian from file.")
    H = Operator.load(args.input_operator)
    H /= 1000 # meV -> eV
    logger.debug("Hamiltonian loaded successfully.")

    # Create spin operators
    logger.info("Creating spin operators.")
    spin_values = np.full(args.N, args.S)
    SpinOp = SpinOperators(spin_values)
    logger.debug(f"Spin values: {spin_values}")

    # Compute magnetic moment operators
    logger.info("Computing magnetic moment operators (Mx, My, Mz).")
    Mx, My, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)

    # Operator selection
    op_map = {"Mx": Mx, "My": My, "Mz": Mz}
    A = op_map[args.operators[0]]
    B = op_map[args.operators[1]]

    # Generate temperature array
    logger.info("Generating temperature array.")
    temp = np.linspace(args.Tmin, args.Tmax, args.Tpoints)
    logger.debug(f"Temperature points: {temp}")

    # Compute susceptibility
    logger.info("Computing susceptibility.")
    sus = susceptibility(temp, H, A, B)
    logger.debug(f"Susceptibility shape: {sus.shape}")

    # Save results
    logger.info(f"Saving susceptibility results to {args.output}.")
    logger.info(f"Saving susceptibility results as CSV to {args.output}.")
    df = pd.DataFrame({
        "temp": temp,
        "sus": sus
    })
    df.to_csv(args.output, index=False)
    logger.info("CSV file written successfully.")

if __name__ == "__main__":
    main()
