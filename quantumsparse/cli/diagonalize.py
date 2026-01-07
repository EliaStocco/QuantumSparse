#!/usr/bin/env python3
import argparse
from quantumsparse.operator import Operator

def main(args):
    
    print("\n=== Diagonalize an operator ===\n")
    
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
    parser = argparse.ArgumentParser(description="Diagonalize the shift operator.")
    parser.add_argument("-i", "--input"   , type=str, required=True , help="pickle input file.")
    parser.add_argument("-o", "--output"  , type=str, required=False, help="pickle output file.", default=None)
    args = parser.parse_args()
    main(args)