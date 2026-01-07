import argparse
import json
import numpy as np
from quantumsparse.operator import Operator

def main():
    
    description = "Summarize the operator information."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-io", "--input_operator"   , type=str, required=True , help="pickle input file with the operator.")
    parser.add_argument("-p", "--plot"    , type=str, required=False, help="output plot file (default: %(default)s).", default=None)
    parser.add_argument("-o", "--options" , type=str, required=False, help="JSON formatted options (default: %(default)s).", default=None)
    parser.add_argument("-e", "--eigenvalues" , type=str, required=False, help="txt file where to save the eigenvalues (default: %(default)s).", default=None)
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    print("\nSummary:")
    print(repr(H))
    
    if args.plot is not None:
        options = json.loads(args.options) if args.options is not None else {}
        options["file"] = args.plot
        print(f"Plotting operator to file '{args.plot}' ... ", end="")
        print(options)
        H.visualize(**options)
        print("done.")
        
    if args.eigenvalues is not None:
        assert H.is_diagonalized(), "The operator must be diagonalized to extract the eigenvalues."
        H = H.sort()
        print(f"Saving eigenvalues to file '{args.eigenvalues}' ... ", end="")
        w = H.eigenvalues.astype(np.complex128)
        np.savetxt(args.eigenvalues, w)
        print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("operator_summary")