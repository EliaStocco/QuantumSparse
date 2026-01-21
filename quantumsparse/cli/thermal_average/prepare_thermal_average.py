import argparse
import pandas as pd
import numpy as np
from quantumsparse.operator import Operator
from quantumsparse.tools.quantum_mechanics import expectation_value

def compute_preparation(H: Operator,Ma: Operator,show=True)->pd.DataFrame:
    if show: print(f"Computing expecation values of the operator A on the eigenstates ... ",end="")
    expA = expectation_value(Ma,H.eigenstates)
    assert np.allclose(expA.imag,0.), "The expectation value should be real."
    if show: print("done.")
    
    if show: print(f"Computing expecation values of the operator A squared on the eigenstates ... ",end="")
    expA2 = expectation_value(Ma@Ma,H.eigenstates)
    assert np.allclose(expA2.imag,0.), "The expectation value should be real."
    if show: print("done.")
    
    if show: print("Preparing results ... ",end="")
    df = pd.DataFrame(columns=["eigenvalues","A","A2"])
    df["eigenvalues"] = H.eigenvalues.real
    df["A"] = expA.real
    df["A2"] = expA2.real
    if show: print("done.")
    
    return df

def main():
    
    description = "Prepare the calculation of the thermal average of a operator A."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-io", "--input_operator"       , type=str, required=True , help="pickle file with the operator.")
    parser.add_argument("-ih", "--input_hamiltonian", type=str, required=True , help="pickle file with the Hamiltonian.")
    parser.add_argument("-o", "--output"         , type=str, required=True, help="csv output file with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading Hamiltonian operator from file '{args.input_hamiltonian}' ... ", end="")
    H = Operator.load(args.input_hamiltonian)
    print("done.")
    
    assert H.is_diagonalized(), "Hamiltonian should be diagonalized."
    assert H.is_hermitean(), "The operator is not hermitean"
    assert np.allclose(H.eigenvalues.imag,0.), "The eigenvalues of the Hamiltonian should be real."
    
    print(f"Reading operator A from file '{args.input_operator}' ... ", end="")
    Ma = Operator.load(args.input_operator)
    print("done.")
    
    df = compute_preparation(H,Ma)
    
    print(f"Saving results to file '{args.output}' ... ",end="")
    df.to_csv(args.output,index=False)
    print("done.")    
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("prepare_thermal_average")