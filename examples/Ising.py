from QuantumSparse.spin.spin_operators import spin_operators
import numpy as np
from QuantumSparse.operator import Operator
from QuantumSparse.spin.interactions import Heisenberg, Ising
from scipy import sparse

def main():
   
    S     = 1./2
    NSpin = 4
    spin_values = np.full(NSpin,S)

    spins = spin_operators(spin_values)

    H = Heisenberg(spins=spins)
    print(H.shape)
    print(H.sparsity())
    # print(H.todense())

    eigenvalues, eigenstates = H.diagonalize()

    # H = Ising(spins.Sz)
   
    print("\n\tJob done :)\n")


if __name__ == "__main__":
   main()