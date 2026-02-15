import argparse
import pandas as pd
import matplotlib.pyplot as plt
from quantumsparse.tools.plot import use_default_style
use_default_style()

def main():
    
    description = "Plot a thermal average vs the temperature."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input" , type=str, required=True , help="csv input file produced by 'compute_susceptibility.py'.")
    parser.add_argument("-o", "--output", type=str, required=True, help="output plot file.")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading results from file '{args.input}' ... ", end="")
    df = pd.read_csv(args.input)
    print("done.")
    
    fluctuation = (df["fluctuation"] - df["average"]**2).sqrt() 
    
    fig, ax = plt.subplots()
    ax.plot(df["temp"], df["average"])
    ax.fill_between(df["temp"], df["average"] - fluctuation, df["average"] + fluctuation, alpha=0.3)
    ax.set_ylabel(r"$<\hat{A}>$")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    
        
    print("Job done :)\n")
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("plot_thermal_average")