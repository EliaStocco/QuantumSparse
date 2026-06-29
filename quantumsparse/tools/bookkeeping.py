# please try not to import quantumsparse here
from typing import Union, TypeVar
ImplErr = ValueError("not implemented yet")

scalar = Union[float,complex]

float_format = "%20.12e"

# Define a generic type variable T
T = TypeVar('T')

TOLERANCE = 1e-8
NOISE = 1e-12

