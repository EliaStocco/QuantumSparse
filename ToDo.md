# To Do list

 - create scripts to:
    - saves Sx, Sy, and Sz to file
    - rotate an operator to cylindrical coordinates
    - build an hamiltonian given a JSON file and the Sx, Sy, and Sz operators (optional)
    - plot the spectrum
    - compares two operators
    - computes the zz magnetic susceptibility
 
 - workflow:
    - saves Sx, Sy, and Sz to file given N and S
    - (optional) rotate Sx, Sy, and Sz onto cylindrical coordinates
    - build the hamiltonian given the interactions in a JSON file and the spin operators
    - (optional) rotate the hamiltonian onto cylindrical coordinates
    - diagonalize the hamiltonian
    - post-process

 - possibilities:
    - cartesian spins, cartesian symmetry, cartesian hamiltonian, diagonalize, rotate
    - cartesian spins, cartesian symmetry, rotate symmetry, cartesian hamiltonian, rotate hamiltonian, diagonalize
    - cartesian spins, rotate spins, cartesian symmetry, rotate symmetry, cylindrical hamiltonian, diagonalize
