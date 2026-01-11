# please try not to import quantumsparse here
from typing import Union, TypeVar
ImplErr = ValueError("not implemented yet")

scalar = Union[float,complex]

# Define a generic type variable T
T = TypeVar('T')

TOLERANCE = 1e-10
NOISE = 1e-12

import argparse
def str2bool(v:Union[bool,str]):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")