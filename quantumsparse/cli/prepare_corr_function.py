import argparse
import pandas as pd
import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.tools.quantum_mechanics import expectation_value

def main():
    
    description = "Prepare the calculation of the a thermal correlation function between two operators A and B."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-ia", "--input_a"   , type=str, required=True , help="pickle file with the first operator.")
    parser.add_argument("-ib", "--input_b"   , type=str, required=False , help="pickle file with the second operator (default: same as 'input_a').", default=None)
    parser.add_argument("-io", "--input_operator"   , type=str, required=True , help="pickle input file with the Hamiltonian operator.")
    parser.add_argument("-o", "--output"      , type=str, required=True, help="csv output file with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading Hamiltonian operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    assert H.is_diagonalized(), "Hamiltonian should be diagonalized."
    assert H.is_hermitean(), "The operator is not hermitean"
    assert np.allclose(H.eigenvalues.imag,0.), "The eigenvalues of the Hamiltonian should be real."
    
    print(f"Reading operator A from file '{args.input_a}' ... ", end="")
    Ma = Operator.load(args.input_a)
    print("done.")
    
    if args.input_b is not None:
        print(f"Reading operator B from file '{args.input_b}' ... ", end="")
        Mb = Operator.load(args.input_b)
        print("done.")
    else:
        Mb = None
    
    print(f"Computing expecation values of the operator A on the eigenstates ... ",end="")
    expA = expectation_value(Ma,H.eigenstates)
    assert np.allclose(expA.imag,0.), "The expectation value should be real."
    print("done.")
    
    if Mb is None:
        expB = expA
        print(f"Computing expecation values of the operator A squared on the eigenstates ... ",end="")
        expAB = expectation_value(Ma@Ma,H.eigenstates)
        assert np.allclose(expAB.imag,0.), "The expectation value should be real."
        print("done.")
    else:
        print(f"Computing expecation values of the operator B on the eigenstates ... ",end="")
        expB = expectation_value(Ma,H.eigenstates)
        assert np.allclose(expB.imag,0.), "The expectation value should be real."
        print("done.")
        
        print(f"Computing expecation values of the operator A@B squared on the eigenstates ... ",end="")
        expAB = expectation_value(Ma@Mb,H.eigenstates)
        assert np.allclose(expAB.imag,0.), "The expectation value should be real."
        print("done.")
        
    print("Preparing results ... ",end="")
    df = pd.DataFrame(columns=["eigenvalues","A","B","AB"])
    df["eigenvalues"] = H.eigenvalues.real
    df["A"] = expA.real
    df["B"] = expB.real
    df["AB"] = expAB.real
    print("done.")
    
    print(f"Saving results to file '{args.output}' ... ",end="")
    df.to_csv(args.output,index=False)
    print("done.")    
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("prepare_corr_function")