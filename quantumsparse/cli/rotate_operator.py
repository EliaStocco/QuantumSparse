import argparse
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator
from quantumsparse.spin.functions import cylindrical_coordinates

def main():
    
    description = "Diagonalize an operator."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-io", "--input_operator"   , type=str, required=True , help="pickle input file with the operator.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="pickle output file with the operator.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")
    
    print("Preparing cylindrical coordinates ... ", end="")
    U = cylindrical_coordinates(SpinOp.spins)
    print("done.")
    
    print(f"Reading operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    print("\nSummary:")
    print(repr(H))
    
    print("Rotating operator to cylindrical basis ... ", end="")
    H = H.unitary_transformation(U)
    print("done.")
    
    print("\nSummary:")
    print(repr(H))
    
    print(f"Saving operator to file '{args.output}' ... ", end="")
    H.save(args.output)
    print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("diagonalize")