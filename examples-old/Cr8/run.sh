#!/bin/bash

# redirect all stdout and stderr for the whole script
exec > log.txt 2> err.txt

# activate your Python environment
pyenv activate qs

# parameters
S="1"
N="4"
folder=".qs-N=${N}-S=${S}"
cylfolder=".qs-N=${N}-S=${S}-cyl"
data="Cr8-U.json"
# MEMSAVE="-m false"

# run commands
mkdir -p tmp eigenvalues sus sus-stable Mz weights

prepare_spins         -N ${N} -S ${S} -o ${folder}
spins_summary         -is ${folder} # optional
prepare_shift         -is ${folder} -o tmp/D.cart.pickle
test_diagonalization  -io tmp/D.cart.pickle
generate_hermitean_operators -is ${folder} -N -1
get_temperatures -T1 0.1 -T2 500 -n 100 -s logspace -o temperatures.txt # -p temperatrues.pdf
compute_magnetic_moments  -is ${folder} -o ${folder}
rotate_spins          -is ${folder} -o ${cylfolder}
compute_magnetic_moments  -is ${cylfolder} -o ${cylfolder}
rotate_operator       -is ${folder} -io ${folder}/Mz.pickle -o ${folder}/Mz.U.pickle

# final
prefix="final-with-sym-U"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle -s tmp/D.cart.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
rotate_operator       -is ${folder} -io H.pickle -o H.pickle
test_diagonalization  -io H.pickle
statistical_weights   -ih H.pickle -t temperatures.txt -o weights/weights.${prefix}.csv
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# final
prefix="final-with-sym-cart"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle -s tmp/D.cart.pickle
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
statistical_weights   -ih H.pickle -t temperatures.txt -o weights/weights.${prefix}.csv
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# final
prefix="final-wo-sym-U"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle #-s tmp/D.cart.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
rotate_operator       -is ${folder} -io H.pickle -o H.pickle
test_diagonalization  -io H.pickle
statistical_weights   -ih H.pickle -t temperatures.txt -o weights/weights.${prefix}.csv
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# final
prefix="final-wo-sym-cart"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle # -s tmp/D.cart.pickle
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
statistical_weights   -ih H.pickle -t temperatures.txt -o weights/weights.${prefix}.csv
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle


# spin in cartesian coordinates --> spin in cylindrical coordinates --> Hamiltonian construction --> diagonalization without symmetries
prefix="spin-cyl-H-diag-wo-sym"
construct_hamiltonian -is ${cylfolder} -j ${data} -o tmp/H.spin-cyl-H.pickle
diagonalize           -io tmp/H.spin-cyl-H.pickle -o H.${prefix}.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization without symmetries
prefix="spin-H-diag-wo-sym-Mz-cart"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization with symmetries
prefix="spin-H-diag-with-sym-Mz-cart"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle -s tmp/D.cart.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization without symmetries --> rotate Mz
prefix="spin-H-diag-wo-sym-Mz-R"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${cylfolder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${cylfolder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization without symmetries --> rotate Mz
prefix="spin-H-diag-wo-sym-Mz-U"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${folder}/Mz.U.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.U.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization with symmetries --> rotate Mz
prefix="spin-H-diag-with-sym-Mz-R"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle -s tmp/D.cart.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${cylfolder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${cylfolder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization with symmetries --> rotate Mz
prefix="spin-H-diag-with-sym-Mz-U"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o H.${prefix}.pickle -s tmp/D.cart.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${folder}/Mz.U.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.U.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle


# # # spin in cartesian coordinates --> Hamiltonian construction + shift --> diagonalization with symmetries --> unitary transformation to cylindrical coordinates
# # ERROR HERE
# tmpprefix="spin-H-diag-with-sym"
# prefix="spin-H-diag-with-sym-U"
# construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
# diagonalize           -io tmp/H.spin-H.pickle -o tmp/H.${tmpprefix}.pickle -s tmp/D.cart.pickle # ${MEMSAVE}
# test_diagonalization  -io tmp/H.${tmpprefix}.pickle
# rotate_operator       -is ${folder} -io tmp/H.${tmpprefix}.pickle -o H.${prefix}.pickle
# ln -s H.${prefix}.pickle H.pickle
# test_diagonalization  -io H.pickle
# operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
# test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
# prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
# compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
# compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
# rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction -->  unitary transformation to cylindrical coordinates --> diagonalization without symmetries
prefix="spin-H-U-diag-wo-sym"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
rotate_operator       -is ${folder} -io tmp/H.spin-H.pickle -o tmp/H.spin-H-U.pickle
diagonalize           -io tmp/H.spin-H-U.pickle -o H.${prefix}.pickle # ${MEMSAVE}
ln -s H.${prefix}.pickle H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
test_diagonalization  -io H.pickle
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# spin in cartesian coordinates --> Hamiltonian construction --> diagonalization without symmetries --> unitary transformation to cylindrical coordinates
prefix="spin-H-diag-wo-sym"
construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
diagonalize           -io tmp/H.spin-H.pickle -o tmp/H.${prefix}.pickle # ${MEMSAVE}
test_diagonalization  -io tmp/H.${prefix}.pickle
rotate_operator       -is ${folder} -io tmp/H.${prefix}.pickle -o H.${prefix}.pickle
ln -s H.${prefix}.pickle H.pickle
test_diagonalization  -io H.pickle
operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
rm H.pickle

# # spin in cartesian coordinates --> Hamiltonian construction + shift -->  unitary transformation to cylindrical coordinates --> diagonalization with symmetries
# # ERROR HERE
# prefix="spin-H-U-diag-with-sym"
# construct_hamiltonian -is ${folder} -j ${data} -o tmp/H.spin-H.pickle
# rotate_operator       -is ${folder} -io tmp/H.spin-H.pickle -o tmp/H.spin-H-U.pickle
# rotate_operator       -is ${folder} -io tmp/D.cart.pickle -o tmp/D.cyl.pickle
# diagonalize           -io tmp/H.spin-H-U.pickle -o H.${prefix}.pickle -s tmp/D.cyl.pickle # ${MEMSAVE}
# ln -s H.${prefix}.pickle H.pickle
# operator_summary      -io H.pickle -e eigenvalues/${prefix}.txt
# test_eigenstates      -is ${folder} -io H.pickle -o tmp/test.${prefix}.csv
# test_diagonalization  -io H.pickle
# prepare_corr_function -ia ${folder}/Mz.pickle -io H.pickle -o Mz/Mz-corr.${prefix}.csv
# compute_susceptibility -i Mz/Mz-corr.${prefix}.csv -t temperatures.txt -o sus/sus.${prefix}.csv
# compute_susceptibility_numerical_stable -ih H.pickle -ia ${folder}/Mz.pickle -t temperatures.txt -o sus-stable/sus-stable.${prefix}.csv
# rm H.pickle
