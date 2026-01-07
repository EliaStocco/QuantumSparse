import numpy as np
from scipy.sparse.linalg import eigsh
A = np.eye(6)
num_eigenvalues = 3
v0 = np.random.rand(A.shape[0], 2)
eigenvalues, eigenvectors = eigsh(A, k=num_eigenvalues, which='SA', v0=v0)

# Exception has occurred: ValueError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
# failed in converting 10th argument `workd' of _arpack.dsaupd to C/Fortran array
#   File "/home/stoccoel/miniconda3/envs/elia/lib/python3.10/site-packages/scipy/sparse/linalg/_eigen/arpack/arpack.py", line 537, in iterate
#     self._arpack_solver(self.ido, self.bmat, self.which, self.k,
#   File "/home/stoccoel/miniconda3/envs/elia/lib/python3.10/site-packages/scipy/sparse/linalg/_eigen/arpack/arpack.py", line 1697, in eigsh
#     params.iterate()
#   File "/home/stoccoel/google-personal/quantumsparse/examples/V8/hello.py", line 12, in <module>
#     eigenvalues, eigenvectors = eigsh(A, k=num_eigenvalues, which='SA', v0=v0)
#   File "/home/stoccoel/miniconda3/envs/elia/lib/python3.10/runpy.py", line 86, in _run_code
#     exec(code, run_globals)
#   File "/home/stoccoel/miniconda3/envs/elia/lib/python3.10/runpy.py", line 196, in _run_module_as_main (Current frame)
#     return _run_code(code, main_globals, None,
# ValueError: failed in converting 10th argument `workd' of _arpack.dsaupd to C/Fortran array

