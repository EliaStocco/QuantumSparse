import argparse
from quantumsparse.operator import Operator

def main():
    
    description = "Diagonalize an operator."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input"   , type=str, required=True , help="pickle input file.")
    parser.add_argument("-o", "--output"  , type=str, required=False, help="pickle output file (default: %(default)s).", default=None)
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading operator from file '{args.input}' ... ", end="")
    H = Operator.load(args.input)
    print("done.")
    
    print("\nSummary:")
    print(repr(H))
    
    H.diagonalize()
    
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