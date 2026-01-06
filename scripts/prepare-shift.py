import argparse
import numpy as np
from quantumsparse.operator import Symmetry
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift
from quantumsparse.conftest import NS2Ops

def main(args):
    
    print("\n=== Diagonalize the shift operator ===\n")
    
    N = args.number
    S = args.spin

    # spin operators
    print(f"Creating spin operators for N={N} spins S={S} ... ", end="")
    Sx, Sy, Sz, SpinOp = NS2Ops(N, S)
    print("done.")
    
    print(f"The dimension of the Hilbert space is {Sx[0].shape[0]}.")

    # symmetry operator (shift)
    print("Creating shift symmetry operator ... ", end="")
    D: Symmetry = shift(SpinOp)
    print("done.")
    
    # print("Diagonalizing shift symmetry operator ... ", end="")
    D.diagonalize()
    # print("done.")
    
    print("\nSummary:")
    print(repr(D))
    
    print("Checking consistency of the energy levels of the shift operator ... ", end="")
    l, n = D.energy_levels()
    assert len(l) == N, "wrong number of energy levels for shift symmetry"
    assert np.allclose(np.sort(l), np.sort(roots_of_unity(N))), "The eigenvalues should be the roots of unity."
    print("done.")
    
    print(f"Saving shift operator to '{args.output}' ... ", end="")
    D.save(args.output)
    print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagonalize the shift operator.")
    parser.add_argument("-N", "--number", type=int  , required=True, help="Number of spins sites in the chain.")
    parser.add_argument("-S", "--spin"  , type=float, required=True, help="Spin quantum number.")
    parser.add_argument("-o", "--output", type=str  , required=True, help="pickle output file to save the operator.")
    args = parser.parse_args()
    main(args)