import argparse
import os
import pandas as pd
import numpy as np
import warnings
from quantumsparse.tools.bookkeeping import TOLERANCE
from quantumsparse.spin import SpinOperators
from quantumsparse.operator import Operator, Symmetry
from quantumsparse.spin.shift import shift
from quantumsparse.tools.quantum_mechanics import expectation_value

def compute_expectation_dataframe(H:Operator, Ma:Operator, Mb:Operator=None)->pd.DataFrame:
    """
    Compute <A>, <B>, and <A B†> on all eigenstates.
    Returns a pandas DataFrame.
    """

    expA = expectation_value(Ma, H.eigenstates)
    assert np.allclose(expA.imag, 0.0), "<A> should be real."

    if Mb is None:
        expB = expA.conjugate()

        expAB = expectation_value(Ma @ Ma.dagger(), H.eigenstates)
        assert np.allclose(expAB.imag, 0.0), "<A A†> should be real."

    else:
        expB = expectation_value(Mb.dagger(), H.eigenstates)
        assert np.allclose(expB.imag, 0.0), "<B> should be real."

        expAB = expectation_value(Ma @ Mb.dagger(), H.eigenstates)
        # assert np.allclose(expAB.imag, 0.0), "<A B†> should be real."

    return pd.DataFrame({
        "eigenvalues": H.eigenvalues.real,
        "A": expA.real,
        "B": expB.real,
        "AB": expAB.real,
    })

def main():
    
    description = "Prepare the correlation function between spin operators (assumes translational invariance)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-is", "--input_spins"   , type=str, required=True , help="folder with the spin information.")
    parser.add_argument("-io", "--input_operator", type=str, required=True , help="pickle input file with the operator.")
    parser.add_argument("-o", "--output"         , type=str, required=True, help="output folder with the results.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading Hamiltonian operator from file '{args.input_operator}' ... ", end="")
    H = Operator.load(args.input_operator)
    print("done.")
    
    assert H.is_diagonalized(), "Hamiltonian should be diagonalized."
    assert H.is_hermitean(), "The operator is not hermitean"
    assert np.allclose(H.eigenvalues.imag,0.), "The eigenvalues of the Hamiltonian should be real."
    
    print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
    SpinOp = SpinOperators.load(args.input_spins)
    print("done.")
    
    print("Constructing shift operator ... ", end="")
    D: Symmetry = shift(SpinOp)
    print("done.")
    
    comm = H.commutator(D).norm()
    if comm > TOLERANCE:
        warnings.warn(
                f"Hamiltonian is not translational invariant: |[H,T]| = {comm}",
                RuntimeWarning
            )

    N = SpinOp.nsites
    os.makedirs(args.output,exist_ok=True)   
    
    info = {
        "x" : SpinOp.Sx, # List[Operator]
        "y" : SpinOp.Sy, # List[Operator]
        "z" : SpinOp.Sz # List[Operator]
    }
    
    results = []

    DEBUG = False

    results = []

    for k1 in ["x", "y", "z"]:
        for k2 in ["x", "y", "z"]:

            if DEBUG:

                reference = {}
                for i in range(N):
                    for j in range(N):
                        r = (j - i) % N
                        df = compute_expectation_dataframe(
                            H,
                            info[k1][i],
                            info[k2][j]
                        )
                        if r not in reference:
                            reference[r] = df
                        else:
                            assert np.allclose(
                                df["AB"],
                                reference[r]["AB"]
                            ), (
                                f"Translation invariance failed "
                                f"for {k1}{k2}, distance {r}, "
                                f"pair ({i},{j})"
                            )

            else:

                for r in range(N):
                    i = 0
                    j = r

                    df = compute_expectation_dataframe(
                        H,
                        info[k1][i],
                        info[k2][j]
                    )

                    file = os.path.join(
                        args.output,
                        f"{k1}{k2}_r{r}.csv"
                    )

                    df.to_csv(file, index=False)

                    results.append({
                        "component_A": k1,
                        "component_B": k2,
                        "distance": r,
                        "site_A": i,
                        "site_B": j,
                        "file": os.path.basename(file),
                    })
    
        pd.DataFrame(results).to_csv(
            os.path.join(args.output, "index.csv"),
            index=False,
        )
                
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("prepare_corr_function_spins")