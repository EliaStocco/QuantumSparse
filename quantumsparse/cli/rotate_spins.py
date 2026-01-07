import argparse
from quantumsparse.spin import SpinOperators
from quantumsparse.spin.functions import get_Euler_angles, rotate_spins
from quantumsparse.tools.debug import check_commutation_relations

def main():
    
    parser = argparse.ArgumentParser(description="Diagonalize the shift operator.")
    parser.add_argument("-i", "--input"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-o", "--output", type=str  , required=True, help="folder with the rotated spin information.")
    parser.add_argument("-m", "--method"   , type=str, required=False , help="folder with the spin information (default: %(default)s).", default="R")
    args = parser.parse_args()
    
    print("\n=== Diagonalize the shift operator ===\n")
    
    print(f"Reading spins from folder '{args.input}' ... ", end="")
    SpinOp = SpinOperators.load(args.input)
    print("done.")
    
    check_commutation_relations(*SpinOp.spins, tolerance=1e-10)
    
    N = SpinOp.nsites
    EulerAngles = get_Euler_angles(N)
    StR, SrR, SzR = rotate_spins(SpinOp.spins, EulerAngles=EulerAngles, method=args.method)
    
    SpinOp.Sx = StR
    SpinOp.Sy = SrR
    SpinOp.Sz = SzR
    
    check_commutation_relations(*SpinOp.spins, tolerance=1e-10)
    
    print(f"Saving shift operator to '{args.output}' ... ", end="")
    SpinOp.save(args.output)
    print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("rotate_spins")