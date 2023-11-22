import glob
import os
import numpy as np
import pandas as pd
from functions import get_couplings
from QuantumSparse.operator import operator
from QuantumSparse.spin.spin_operators import spin_operators
from QuantumSparse.spin.functions import magnetic_moments, rotate_spins
from QuantumSparse.spin.interactions import Heisenberg, DM, anisotropy, rhombicity, Ising

import argparse

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description="Your script description here")
parser.add_argument("--restart", type=bool, default=True, help="Set to True if restarting from a previous run.")
parser.add_argument("--NSpin", type=int, default=4, help="Number of spins.")
parser.add_argument("--S", type=float, default=1., help="Value of S.")
parser.add_argument("--datafolder", type=str, default="data", help="Path to the data folder.")
parser.add_argument("--output_folder", type=str, default="output", help="Path to the output folder.")
parser.add_argument("--name", type=str, default="V8", help="Base name for files.")
parser.add_argument("--tol", type=float, default=1e-6, help="tolerance")
parser.add_argument("--diagtol", type=float, default=1., help="initial diagonalization tolerance")
parser.add_argument("--increment", type=float, default=3.0, help="initial diagonalization tolerance")
args = parser.parse_args()

def getfile(folder):
    # Check if the output folder exists
    if os.path.exists(folder):
        # Get a list of all .npz files in the folder that match the naming convention
        npz_files = glob.glob(os.path.join(folder, "Hamiltonian.step=*.npz"))

        if npz_files:
            max_step = -1  # Initialize the maximum step as -1
            max_step_file = None

            for file in npz_files:
                # Extract the step number from the file name
                step = int(file.split("Hamiltonian.step=")[1].split(".npz")[0])

                # Check if this file has a larger step
                if step > max_step:
                    max_step = step
                    max_step_file = file

            if max_step_file is not None:
                print(f"The file with the largest step is: {max_step_file}")
                print(f"Step: {max_step}")
                return os.path.normpath("{:s}/Hamiltonian.step={:d}.npz".format(folder,max_step))
            else:
                print("No valid .npz files found in the 'output' folder.")
                return None
        else:
            print("No .npz files found in the 'output' folder that match the naming convention.")
            return None
    else:
        print(f"The '{folder}' folder does not exist.")
        return None
    
def buildH(S,NSpin,folder,name)->operator:

    # build spin operators
    #print("\tbuilding spin operators ... ",end="")
    # S     = 1
    # NSpin = 8
    spin_values = np.full(NSpin,S)
    spins = spin_operators(spin_values)

    totS2 = spins.compute_total_S2()
    S2 = spins.compute_S2()
    #print("done")

    # rotate spins
    print("\n\tcomputing Euler's angles ... ",end="")
    EulerAngles = np.zeros((NSpin,3))
    EulerAngles[:,2] = 360 - np.linspace(0,360,NSpin,endpoint=False)
    EulerAngles = np.pi * EulerAngles / 180  
    #EulerAngles.fill(0)
    print("done")

    print("\trotating spins ... ",end="")           
    # St,Sr,Sz= rotate_spins(spins=spins,EulerAngles=EulerAngles)
    # spins.Sx,spins.Sy,spins.Sz = St,Sr,Sz
    St,Sr,Sz = spins.Sx,spins.Sy,spins.Sz
    print("done")

    # load coupling constants
    print("\treading coupling constants ... ",end="")           
    couplings = get_couplings(S,folder,name)
    print("done")

    # build the hamiltonian
    dim = St[0].shape[0]
    print("\tHilbert space dimension: {:d}".format(dim))   
    print("\tbuilding the Hamiltonian: ")           

    # H = operator((dim,dim))
    H = Heisenberg(Sx=St,Sy=Sr,Sz=Sz,couplings=[couplings["Jt"],couplings["Jr"],couplings["Jz"]])
    print("\t\t'Heisenberg 1nn': {:d} blocks".format(H.count_blocks()[0]))     

    H += DM(Sx=St,Sy=Sr,Sz=Sz,couplings=[couplings["dt"],couplings["dr"],couplings["dz"]])
    print("\t\t            'DM': {:d} blocks".format(H.count_blocks()[0]))      

    H += anisotropy(Sz=Sz,couplings=couplings["D"])
    print("\t\t    'anisotropy': {:d} blocks".format(H.count_blocks()[0]))       

    H += rhombicity(Sx=St,Sy=Sr,couplings=couplings["E"])
    print("\t\t    'rhombicity': {:d} blocks".format(H.count_blocks()[0]))       

    H += Heisenberg(Sx=St,Sy=Sr,Sz=Sz,nn=2,couplings=[couplings["Jt2"],couplings["Jr2"],couplings["Jz2"]])
    print("\t\t'Heisenberg 2nn': {:d} blocks".format(H.count_blocks()[0]))       

    # H  = H * 1E-3 # data are in meV, we build the Hamiltonian in eV
    print("done")

    return H, spins
   
folder = args.output_folder
file = getfile(folder)
if file is None or args.restart:
    S = args.S
    NSpin = args.NSpin 
    datafolder = args.datafolder
    name = args.name
    H, spins = buildH(S,NSpin,datafolder,name)
else :
    H = operator.load(file)

from QuantumSparse.spin.shift import shift
from QuantumSparse.spin.flip import flip
D = shift(spins)
D.diagonalize(method="jacobi")

F = flip(spins)
F.diagonalize(method="jacobi")

# Sz = spins.Sz.sum()
# Sz.diagonalize(method="jacobi")

H.diagonalize_with_symmetry(S=[D,F],tol=1e-4)
H.diagonalize_with_symmetry(S=[D,F],tol=1e-5)

# H.visualize(file="V8.png")
# D.visualize(file="D.png")
# Create an empty DataFrame
data = pd.DataFrame(columns=["step", "test", "file","diagtol"])

if not os.path.exists(folder): 
    os.mkdir(folder)

k = 0
test = np.inf
tol = args.tol
diagtol = args.diagtol
while test > tol :

    print("\tDiagonalizing: step {:d} with tolerance {:.6e}".format(k,tol))

    # diagonalize
    w,f = H.diagonalize(method="jacobi",tol=diagtol)

    # test diagonalization
    _,test = H.test_diagonalization(tol=tol,return_norm=True)

    # save to file
    file = os.path.normpath("{:s}/Hamiltonian.step={:d}.npz".format(folder,k+1))
    print("\tsaving results to file '{:s}'".format(file))
    H.save(file)

    data = data.append({"step": k + 1, "test": test, "file": file,"diagtol":diagtol}, ignore_index=True)
    data.to_csv("{:s}/report.csv".format(folder),index=False)

    # update parameters
    k += 1
    diagtol /= 10.

print("\n\tJob done :)\n")

