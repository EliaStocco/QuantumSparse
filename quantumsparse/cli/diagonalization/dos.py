import argparse
import numpy as np
from quantumsparse.tools.functions import lorentzian
from quantumsparse.tools.bookkeeping import float_format

def main():
    
    description = "Compute the DOS from the eigenvalues as a sum of Lorentzians."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-e", "--eigenvalues"   , type=str, required=True , help="txt input file with the eigenvalues.")
    parser.add_argument("-l", "--energy_lims", type=float, nargs=2, required=False, help="energy limits for the DOS (default: %(default)s).", default=None)
    parser.add_argument("-N", "--number", type=int, required=False, help="number of points for the DOS (default: %(default)s).", default=10000)
    parser.add_argument("-w", "--gamma"      , type=str, required=False, help="width of the Lorentzians (default: %(default)s).", default=None)
    parser.add_argument("-o", "--output"   , type=str, required=True , help="txt output file with the DOS.")
    parser.add_argument("-p", "--plot"   , type=str, required=False , help="output plot file with the DOS (default: %(default)s).",default=None)
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")
    
    print(f"Reading data from file '{args.eigenvalues}' ... ",end="")
    eigenvalues = np.loadtxt(args.eigenvalues,dtype=complex)
    print("done")
    assert np.allclose(eigenvalues.imag,0), "The imaginary part is not zero."
    eigenvalues = eigenvalues.real
    
    if args.energy_lims is None:
        emin = eigenvalues.min()
        emax = eigenvalues.max()
        width = emax-emin
        delta = width/12.
        args.energy_lims = [emin-delta,emax+delta]
        
    print("Using the following energy limits: ",*args.energy_lims)
        
    if args.gamma is None:
        width = args.energy_lims[1] - args.energy_lims[0]
        args.gamma = width/(12.*100)
    print("Using the following width for the Lorentzians: ",args.gamma)
    
    energy = np.linspace(args.energy_lims[0],args.energy_lims[1],args.number+1,endpoint=True)
    spectrum = np.zeros(args.number+1)
    for x0 in eigenvalues:
        spectrum += lorentzian(x=energy,x0=x0,y0=1,gamma=args.gamma)
    data = np.vstack((energy,spectrum)).T
    
    print(f"Saving data to file '{args.output}' ... ",end="")
    np.savetxt(args.output,data,fmt=float_format,header="energy [eV]          DOS")
    print("done")
    
    if args.plot is not None:
        import matplotlib.pyplot as plt
        from quantumsparse.tools.plot import use_default_style
        use_default_style()
        
        fig, ax = plt.subplots()
        ax.plot(energy,spectrum)
        ax.set_yscale("symlog")
        ax.set_xlabel("energy [eV]")
        ax.set_ylabel("DOS")
        
        print(f"Saving plot to file '{args.plot}' ... ",end="")
        plt.savefig(args.plot,dpi=600) 
        print("done")       
    
    print("Job done :)\n")
    
    
if __name__ == "__main__":
    main()
    
def test_script():
    from quantumsparse.conftest import template_test_script
    template_test_script("dos")