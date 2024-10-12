from typing import TypeVar, Union, List
import numpy as np

# Define a generic type variable T
T = TypeVar('T')

# MyList can now be a list of T or a NumPy array
NDArray = Union[List[T], np.ndarray]