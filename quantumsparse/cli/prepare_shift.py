import argparse
import numpy as np
from quantumsparse.operator import Symmetry
from quantumsparse.spin import SpinOperators
from quantumsparse.tools.mathematics import roots_of_unity
from quantumsparse.spin.shift import shift

def main():
    
    parser = argparse.ArgumentParser(description="Diagonalize the shift operator.")
    parser.add_argument("-i", "--input"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-o", "--output", type=str  , required=True, help="pickle output file to save the operator.")
    args = parser.parse_args()
    
    print("\n=== Diagonalize the shift operator ===\n")
    
    print(f"Reading spins from folder '{args.input}' ... ", end="")
    SpinOp = SpinOperators.load(args.input)
    print("done.")

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
    N = SpinOp.nsites
    assert len(l) == N, "wrong number of energy levels for shift symmetry"
    assert np.allclose(np.sort(l), np.sort(roots_of_unity(N))), "The eigenvalues should be the roots of unity."
    print("done.")
    
    print(f"Saving shift operator to '{args.output}' ... ", end="")
    D.save(args.output)
    print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("prepare_shift")