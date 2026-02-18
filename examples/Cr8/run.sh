#!/bin/bash

# redirect all stdout and stderr for the whole script
exec > log.txt 2> err.txt

# activate your Python environment
pyenv activate qs

# parameters
S="0.5"
N="4"
folder=".qs-N=${N}-S=${S}"
cylfolder=".qs-N=${N}-S=${S}-cyl"
data="Cr8-U.json"
MEMSAVE="-m true"

# run commands
mkdir -p tmp eigenvalues 

prepare_spins         -N ${N} -S ${S} -o ${folder}
spins_summary         -is ${folder} # optional
prepare_shift         -is ${folder} -o tmp/D.cart.pickle
generate_hermitean_operators -is ${folder} -N -1

# spin in cartesian coordinates --> spin in cylindrical coordinates --> Hamiltonian construction --> diagonalization without symmetries
rotate_spins          -is ${folder} -o ${cylfolder}
construct_hamiltonian -is ${cylfolder} -j ${data} -o tmp/H.spin-cyl-H.pickle
diagonalize           -io tmp/H.spin-cyl-H.pickle -o H.spin-cyl-H-diag-wo-sym.pickle ${MEMSAVE}
operator_summary      -io H.spin-cyl-H-diag-wo-sym.pickle -e eigenvalues/spin-cyl-H-diag-wo-sym.txt
test_eigenstates      -is ${folder} -io H.spin-cyl-H-diag-wo-sym.pickle -o tmp/test.spin-cyl-H-diag-wo-sym.csv
test_diagonalization  -io H.spin-cyl-H-diag-wo-sym.pickle

# spin in cartesian coordinates --> Hamiltonian construction -->  unitary transformation to cylindrical coordinates --> diagonalization without symmetries
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
rotate_operator       -is ${folder} -io tmp/H.spin-H.pickle -o tmp/H.spin-H-U.pickle
diagonalize           -io tmp/H.spin-H-U.pickle -o H.spin-H-U-diag-wo-sym.pickle ${MEMSAVE}
operator_summary      -io H.spin-H-U-diag-wo-sym.pickle -e eigenvalues/spin-H-U-diag-wo-sym.txt
test_eigenstates      -is ${folder} -io H.spin-H-U-diag-wo-sym.pickle -o tmp/test.spin-H-U-diag-wo-sym.csv
test_diagonalization  -io H.spin-H-U-diag-wo-sym.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization without symmetries --> unitary transformation to cylindrical coordinates
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o tmp/H.spin-H-diag-wo-sym.pickle ${MEMSAVE}
rotate_operator       -is ${folder} -io tmp/H.spin-H-diag-wo-sym.pickle -o H.spin-H-diag-wo-sym-U.pickle
operator_summary      -io H.spin-H-diag-wo-sym-U.pickle -e eigenvalues/spin-H-diag-wo-sym-U.txt
test_eigenstates      -is ${folder} -io H.spin-H-diag-wo-sym-U.pickle -o tmp/test.spin-H-diag-wo-sym-U.csv
test_diagonalization  -io H.spin-H-diag-wo-sym-U.pickle

# spin in cartesian coordinates --> Hamiltonian construction + shift --> diagonalization with symmetries --> unitary transformation to cylindrical coordinates
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o tmp/H.spin-H-diag-with-sym.pickle -s tmp/D.cart.pickle ${MEMSAVE}
rotate_operator       -is ${folder} -io tmp/H.spin-H-diag-with-sym.pickle -o H.spin-H-diag-with-sym-U.pickle
operator_summary      -io H.spin-H-diag-with-sym-U.pickle -e eigenvalues/spin-H-diag-with-sym-U.txt
test_eigenstates      -is ${folder} -io H.spin-H-diag-with-sym-U.pickle -o tmp/test.spin-H-diag-with-sym-U.csv
test_diagonalization  -io H.spin-H-diag-with-sym-U.pickle

# spin in cartesian coordinates --> Hamiltonian construction + shift -->  unitary transformation to cylindrical coordinates --> diagonalization with symmetries
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
rotate_operator       -is ${folder} -io tmp/H.spin-H.pickle -o tmp/H.spin-H-U.pickle
rotate_operator       -is ${folder} -io tmp/D.cart.pickle -o tmp/D.cyl.pickle
diagonalize           -io tmp/H.spin-H-U.pickle -o H.spin-H-U-diag-with-sym.pickle -s tmp/D.cyl.pickle ${MEMSAVE}
operator_summary      -io H.spin-H-U-diag-with-sym.pickle -e eigenvalues/spin-H-U-diag-with-sym.txt
test_eigenstates      -is ${folder} -io H.spin-H-U-diag-with-sym.pickle -o tmp/test.spin-H-U-diag-with-sym.csv
test_diagonalization  -io H.spin-H-U-diag-with-sym.pickle

