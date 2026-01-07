import argparse
from quantumsparse.operator import Operator, Symmetry

def main():
    
    description = "Diagonalize an operator."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-io", "--input_operator"   , type=str, required=True , help="pickle input file with the operator.")
    parser.add_argument("-s", "--symmetry", type=str, required=False, help="pickle input file of the symmetry operator (default: %(default)s).", default=None)
    parser.add_argument("-o", "--output"      , type=str, required=True, help="pickle output file with the operator.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    print("\nSummary:")
    print(repr(H))
    
    if args.symmetry is not None:
        print(f"\nReading symmetry operator from file '{args.symmetry}' ... ", end="")
        S = Symmetry.load(args.symmetry)
        print("done.")
        
        print("Diagonalizing operator in the symmetry-adapted basis ... ", end="")
        H.diagonalize_with_symmetry(S=S)
        print("done.")
    
    else:
        print("Diagonalizing operator ... ", end="")
        H.diagonalize()
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