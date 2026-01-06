import argparse
import json
from quantumsparse.operator import Operator

def main(args):
    
    print("\n=== Summary of an operator ===\n")
    
    print(f"Reading operator from file '{args.input}' ... ", end="")
    H = Operator.load(args.input)
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
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagonalize the shift operator.")
    parser.add_argument("-i", "--input"   , type=str, required=True , help="pickle input file.")
    parser.add_argument("-p", "--plot"    , type=str, required=False, help="output plot file.", default=None)
    parser.add_argument("-o", "--options" , type=str, required=False, help="JSON formatted options.", default=None)
    args = parser.parse_args()
    main(args)