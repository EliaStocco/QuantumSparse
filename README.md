# QuantumSparse

A `scipy.sparse` based package to represent quantum spin operators and exact diagonalization.

## Installation
You can install this package in *editable* mode using
```bash
pip install -e .
```

## Example
How to efficiently diagonalize a Heisenberg Hamiltonian:
```python
import numpy as np
from quantumsparse.spin import SpinOperators, Heisenberg
from quantumsparse.spin.shift import shift

# Spin value and number of sites
S = 0.5
N = 4

# Create spin operators for N spin-S sites
spin_values = np.full(N, S)
SpinOp = SpinOperators(spin_values)
Sx, Sy, Sz = SpinOp.Sx, SpinOp.Sy, SpinOp.Sz

# Build the Heisenberg Hamiltonian
H = Heisenberg(Sx, Sy, Sz)

# Diagonalize without symmetry
E_no_sym, _ = H.diagonalize()
print("Ground state energy (no symmetry):", E_no_sym[0].real)

# Apply shift symmetry
D = shift(SpinOp)
D.diagonalize()

# Diagonalize using symmetry
E_sym, _ = H.diagonalize_with_symmetry(S=[D])
print("Ground state energy (with shift symmetry):", E_sym[0].real)

H.save("H.pickle")
print("Saved Hamiltonian to file 'H.pickle'")
```