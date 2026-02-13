import argparse
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.tools.bookkeeping import str2bool

def main():
    
    description = "Diagonalize an operator."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-io", "--input_operator"   , type=str, required=True , help="pickle input file with the operator.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    print("\nSummary:")
    print(repr(H))
    
    solution = H.test_eigensolution()
    norm = solution.norm()
    print("\n\teigensolution norm:",norm)
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("test_diagonalization")