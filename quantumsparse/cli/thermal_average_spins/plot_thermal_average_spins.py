import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from quantumsparse.constants import kB
from quantumsparse.tools.plot import use_default_style

use_default_style()

# Secondary x-axis converting T <-> E=kBT
def T_to_E(T):
    return kB * T

def E_to_T(E):
    return E / kB
            
def main():
    
    description = "Plot the expectation value of all spin operators."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="folder filled by 'prepare_thermal_average_spins.py'.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output file with the image.")
    args = parser.parse_args()
        
    print(f"\n=== {description} ===\n")
    
    files = [file for file in os.listdir(args.input)]
    Nsites = int(len(files) / 3)
    
    fig, axes = plt.subplots(Nsites, 3, figsize=(4.5,1.5 * Nsites),
                             sharey=True, sharex=True)
        
    for site in range(Nsites):
        for n, xyz in enumerate(["x", "y", "z"]):
        
            pfile = f"{args.input}/S{xyz}_{site}.csv"
            print(f"Reading results from file '{pfile}' ... ", end="")
            prepare_df = pd.read_csv(pfile)
            print("done.")
            
            prepare_df["eigenvalues"] -= prepare_df["eigenvalues"].min()
            
            ax = axes[site, n]
            ax.set_title(rf"$\langle \hat{{S}}^{{{xyz}}}_{{{site}}}\rangle$")
            
            x = 1000 * prepare_df["eigenvalues"].to_numpy()
            y = prepare_df["A"].to_numpy()

            # Group points to detect degeneracy
            groups = defaultdict(list)
            for xi, yi in zip(x, y):
                key = (np.round(xi, 8), np.round(yi, 8))
                groups[key].append((xi, yi))

            # Scatter plot
            ax.scatter(x, y, color="blue")

            # Annotate degeneracies
            for (xi, yi), pts in groups.items():
                deg = len(pts)
                if deg > 1:
                    ax.text(
                        xi, yi,
                        str(deg),
                        fontsize=8,
                        color="red",
                        ha="center",
                        va="bottom"
                    )
       
    fig.supxlabel('energy eigenvalues [meV]')
    fig.supylabel('spin expectation value')    
    plt.tight_layout()
    plt.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)    
        
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()

def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("plot_thermal_average_spins")