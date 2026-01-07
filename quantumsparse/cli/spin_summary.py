import argparse
from quantumsparse.spin import SpinOperators

def main():
    
    parser = argparse.ArgumentParser(description="Diagonalize the shift operator.")
    parser.add_argument("-i", "--input"   , type=str, required=True , help="folder with the spin information.")
    args = parser.parse_args()
    
    print("\n=== Summary of an operator ===\n")
    
    print(f"Reading spins from folder '{args.input}' ... ", end="")
    SpinOp = SpinOperators.load(args.input)
    print("done.")
    
    print(f"sites: {len(SpinOp.spin_values)}.")
    print(f"spins: {SpinOp.spin_values}.")
    print(f"Hilber space dimension: {SpinOp.Sx[0].shape[0]}.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("spin_summary")