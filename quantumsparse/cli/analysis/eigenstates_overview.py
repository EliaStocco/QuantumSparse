import argparse
import pandas as pd
import numpy as np
import os
from quantumsparse.spin import SpinOperators
from quantumsparse.cli import ilist, str2index, str2bool
from quantumsparse.operator import Operator
from quantumsparse.tools.bookkeeping import float_format

def main():
    
    description = "Provides an overview of the eigenstates."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-ih", "--input_hamiltonian", type=str, required=True , help="pickle file with the Hamiltonian.")
    parser.add_argument("-e", "--energy_limits"   , type=str, required=False , help="energy limits (default: %(default)s).", default=None)
    parser.add_argument("-n", "--indices"   , type=str2index, required=False , help="eigenstate indices (default: %(default)s).", default=None)
    parser.add_argument("-s", "--sort"   , type=str2bool, required=False , help="sort the output (default: %(default)s).", default=True)
    parser.add_argument("-o", "--output"      , type=str, required=True, help="output folder.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    assert not (
        args.energy_limits is None and args.indices is None
    ), "You must specify either '--energy_limits' or '--indices'."

    assert (
        args.energy_limits is None or args.indices is None
    ), "Options '--energy_limits' and '--indices' are mutually exclusive."
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")
    basis = SpinOp.basis
    basis.columns = [f"site={i}" for i in basis.columns]
    basis = basis.reset_index(names="basis state")
    
    print(basis.to_string(index=False))
    print()
    
    print(f"Reading Hamiltonian from file '{args.input_hamiltonian}' ... ", end="")
    H = Operator.load(args.input_hamiltonian)
    H = H.sort()
    print("done.")
    assert H.is_diagonalized(), "Hamiltonian should be diagonalized."
    
    print("Extracting eigenvalues ... ",end="")
    energies = H.eigenvalues[args.indices]
    print("done.")
    print("eigenvalues.shape: ", energies.shape)
    
    print("Extracting eigenstates ... ",end="")
    states = H.eigenstates[args.indices]
    print("done.")
    print("eigenstates.shape: ", states.shape)
    
    os.makedirs(args.output,exist_ok=True)
    for n,(energy,state) in enumerate(zip(energies,states)):
        ofile = f"{args.output}/eigenstate.n={n}.csv"
        print(f" - {n:3}) energy {1000*energy:8.3f} meV --> {ofile}")
        assert state.todense().shape[0] == 1
        state = state.todense().flatten()
        
        df = basis.copy()
        df["real"] = state.real
        df["imag"] = state.imag
        df["abs"] = np.abs(state)
        
        if args.sort:
            df.sort_values(by="abs",ascending=False,inplace=True)
        
        header = f"energy: {1000*energy}  meV"
        df.to_csv(ofile,index=False,header=header)
        pass
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("eigenstate_overview")