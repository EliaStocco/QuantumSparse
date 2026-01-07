import argparse
from quantumsparse.conftest import NS2Ops

def main():
    
    parser = argparse.ArgumentParser(description="Diagonalize the shift operator.")
    parser.add_argument("-N", "--number", type=int  , required=True, help="Number of spins sites in the chain.")
    parser.add_argument("-S", "--spin"  , type=float, required=True, help="Spin quantum number.")
    parser.add_argument("-o", "--output", type=str  , required=True, help="output folder to save the operators.")
    args = parser.parse_args()
    
    print("\n=== Diagonalize the shift operator ===\n")
    
    N = args.number
    S = args.spin

    # spin operators
    print(f"Creating spin operators for N={N} spins S={S} ... ", end="")
    _, _, _, SpinOp = NS2Ops(N, S)
    print("done.")
    
    print(f"The dimension of the Hilbert space is {len(SpinOp.spin_values)}.")

    print(f"Saving shift operator to '{args.output}' ... ", end="")
    SpinOp.save(args.output)
    print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("prepare_spins")