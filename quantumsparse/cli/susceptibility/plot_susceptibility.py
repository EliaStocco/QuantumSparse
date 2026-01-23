import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantumsparse.spin import SpinOperators
from quantumsparse.statistics import Curie_constant
from quantumsparse.tools.plot import use_default_style
use_default_style()

def main():
    
    description = "Plot the magnetic susceptibility vs the temperature."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input" , type=str, required=True , help="csv input file produced by 'compute_susceptibility.py'.")
    parser.add_argument("-is", "--input_spins"   , type=str, required=False, help="folder with the spin information to plot the free-spin susceptibility.", default=None)
    parser.add_argument("-o", "--output", type=str, required=True, help="output plot file.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading results from file '{args.input}' ... ", end="")
    df = pd.read_csv(args.input)
    print("done.")
    
    if args.input_spins is not None:
        print(f"Reading spins from folder '{args.input_spins}' ... ", end="")
        SpinOp = SpinOperators.load(args.input_spins)
        print("done.")  
        C = Curie_constant(SpinOp.spin_values)
    else:
        C = None
    
    fig, axes = plt.subplots(1,3)
    ax = axes[0]
    ax.plot(df["temp"], df["sus"])
    ax.set_ylabel(r"$\chi$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if C is not None:
        sus_vals = np.asarray(C/df["temp"])
        ax.plot(df["temp"], sus_vals, "--")
    
    ax = axes[1]
    ax.plot(df["temp"], df["sus"]*df["temp"])
    ax.set_ylabel(r"$\chi T$")
    ax.set_xlabel("temperature [K]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if C is not None:
        susT_vals = np.full(len(df["temp"]),C)
        ax.plot(df["temp"], susT_vals, "--")
    
    ax = axes[2]
    ax.plot(df["temp"], 1./df["sus"])
    ax.set_ylabel(r"$\chi^{-1}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if C is not None:
        inv_sus_vals = np.asarray(df["temp"]/C)
        ax.plot(df["temp"], inv_sus_vals, "--")
        
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("plot_susceptibility")