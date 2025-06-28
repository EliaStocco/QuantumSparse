import numpy as np
from typing import Union, List
from quantumsparse.operator import Operator
ImplErr = ValueError("not implemented yet")

OpArr = Union[List[Operator],np.ndarray]
scalar = Union[float,complex]