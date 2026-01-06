# please try not to import quantumsparse here
from typing import Union, TypeVar
ImplErr = ValueError("not implemented yet")

scalar = Union[float,complex]

# Define a generic type variable T
T = TypeVar('T')

TOLERANCE = 1e-10
NOISE = 1e-12