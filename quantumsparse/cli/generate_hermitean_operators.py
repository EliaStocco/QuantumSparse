import argparse
from typing import List
from quantumsparse.operator import Operator
from quantumsparse.spin import SpinOperators

def hermitian_basis(d, max_operators=-1)->List[Operator]:
    """
    Generate a full basis of Hermitian operators for a Hilbert space of dimension d.

    Args:
        d: dimension of the Hilbert space
        max_operators: number of operators to generate (default: d*d)

    Returns:
        List of Operator instances forming a Hermitian basis.
    """
    if max_operators < 0:
        max_operators = d * d
    elif max_operators > d * d:
        max_operators = d * d

    # Preallocate
    basis = [None] * max_operators
    count = 0

    # Diagonal elements
    for i in range(d):
        if count >= max_operators:
            return basis
        basis[count] = Operator.one_hot(d, i, i)
        count += 1

    # Off-diagonal elements
    for i in range(d):
        for j in range(i + 1, d):
            if count >= max_operators:
                return basis
            # Symmetric real: |i><j| + |j><i|
            basis[count] = Operator.one_hot(d, i, j) + Operator.one_hot(d, j, i)
            count += 1

            if count >= max_operators:
                return basis
            # Anti-symmetric imaginary: -i(|i><j| - |j><i|)
            basis[count] = -1j * Operator.one_hot(d, i, j) + 1j * Operator.one_hot(d, j, i)
            count += 1

    return basis


def main():
    
    description = "Generate (possibly) all hermitean operators of a Hilbert space."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-N", "--max_N"  , type=str, required=False , help="maximum number of operators to consider (default: %(default)s).", default=100)
    parser.add_argument("-o", "--output_folder"  , type=str, required=False , help="output folder with the hermitean operators (default: %(default)s).", default=None)
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")
    
    if args.output_folder is None:
        args.output_folder = f"{args.input_spins}/hermitean-operators"
        
    print(f"Generating hermitean operators (max {args.max_N}) ... ", end="")
    operators = hermitian_basis(SpinOp.Hilbert_space_dimension,args.max_N)
    print("done.")
    
    print(f"Saving hermitean operators to folder '{args.output_folder}' ... ", end="")
    for i,op in enumerate(operators):
        op.save(f"{args.output_folder}/op_{i}.pickle")
    print("done.")
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("generate_hermitean_operators")