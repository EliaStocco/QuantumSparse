
import sys,os
dir_ = os.path.dirname(__file__)
os.chdir(dir_)
print("\n\tchanging working directory to:",dir_)
sys.path.append("../")
import QuantumSparse as qs
import numpy as np
import matplotlib.pyplot as plt

chain = qs.fermionic_operators(N=2,quantum_numbers={"sites":[0,1], "spin":["up","dw"]})


spin_chain = qs.spin_system(N=2,S=0.5)


masks = chain.masks(quantum_numbers=["spin"])

# chain._template[masks["up"]] = 4
# chain._template[masks["dw"]] = 8


n = chain.number_operators(["spin"])


from scipy import sparse
#from QuantumSparse.operators import quantum_operator
a = qs.quantum_operator(sparse.eye(3))
#a.is_diagonal()