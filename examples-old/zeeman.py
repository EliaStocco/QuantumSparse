#!/usr/bin/env python3
import argparse
import numpy as np
import logging
import os
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import magnetic_moments

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Compute magnetic susceptibility.")
    parser.add_argument("-i", "--input", required=True, help="Input file of H (pickle format).")
    parser.add_argument("-s", "--symmetry", required=True, help="Input file of S (pickle format).")
    parser.add_argument("-o", "--output", required=True, help="Output file prefix (without extension).")
    parser.add_argument("-N", type=int, required=True, help="Number of spin sites.")
    parser.add_argument("-S", type=float, required=True, help="Spin value per site.")
    parser.add_argument("--Bmin", type=float, default=0, help="Minimum magnetic field (default: 0 T).")
    parser.add_argument("--Bmax", type=float, default=1.0, help="Maximum magnetic field (default: 1 T).")
    parser.add_argument("--Bpoints", type=int, default=10, help="Number of magnetic field points (default: 10).")
    parser.add_argument("--method", choices=["linspace", "geomspace"], default="linspace",
                        help="Method for generating magnetic field points.")
    parser.add_argument("--logfile", default="zeeman.log", help="Log file name (default: zeeman.log).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--restart", action="store_true", help="Re-run all magnetic fields (ignore existing files).")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        filename=args.logfile,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger()

    logger.info("Starting Zeeman calculation.")
    logger.info(f"Input Hamiltonian: {args.input}")
    logger.info(f"Symmetry file: {args.symmetry}")
    logger.info(f"Output prefix: {args.output}")
    logger.info(f"Spin value: S = {args.S}, Number of sites: N = {args.N}")
    logger.info(f"Magnetic field: {args.Bmin} to {args.Bmax}, Points: {args.Bpoints}, Method: {args.method}")
    logger.info(f"Restart mode: {args.restart}")
    logger.info(f"Logging to: {args.logfile}")
    if args.debug:
        logger.debug("Debug mode is ON.")

    # Load the Hamiltonian
    logger.info("Loading Hamiltonian from file.")
    H0 = Operator.load(args.input_operator)
    H0.eigenvalues = False
    H0.eigenstates = False
    H0 /= 1000  # meV -> eV
    logger.debug("Hamiltonian loaded successfully.")

    # Load symmetry
    logger.info("Loading symmetry operator from file.")
    S = Symmetry.load(args.symmetry)
    logger.debug("Symmetry operator loaded successfully.")

    # Create spin operators
    logger.info("Creating spin operators.")
    spin_values = np.full(args.N, args.S)
    SpinOp = SpinOperators(spin_values)
    logger.debug(f"Spin values: {spin_values}")

    # Compute magnetic moment operators
    logger.info("Computing magnetic moment operators (Mx, My, Mz).")
    _, _, Mz = magnetic_moments(SpinOp.Sx, SpinOp.Sy, SpinOp.Sz)

    # Generate magnetic field array
    logger.info("Generating magnetic field array.")
    if args.method == "linspace":
        magB = np.linspace(args.Bmin, args.Bmax, args.Bpoints)
    elif args.method == "geomspace":
        magB = np.geomspace(args.Bmin, args.Bmax, args.Bpoints)
    else:
        raise ValueError(f"{args.method} is not supported")
    logger.debug(f"Magnetic field points: {magB}")

    for n, b in enumerate(magB):
        outfile = f"{args.output}.n={n}.pickle"

        # Skip if file exists and not restarting
        if not args.restart and os.path.exists(outfile):
            logger.info(f"Skipping B={b:.6f} T (output {outfile} exists).")
            continue

        logger.info(f"Diagonalization {n+1}/{args.Bpoints} for B={b:.6f} T...")
        H = Operator(H0 - Mz * b)
        H.diagonalize_with_symmetry(S)
        logger.info(f"Finished diagonalization {n+1}/{args.Bpoints}.")

        logger.info(f"Saving Hamiltonian to file {outfile}")
        H.save(outfile)

    logger.info("Calculation finished.")

if __name__ == "__main__":
    main()
