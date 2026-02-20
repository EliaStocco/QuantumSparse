import os
import numpy as np
import warnings
# Turn **all warnings** into exceptions
warnings.filterwarnings("error")

def get_dtype():
    dtype_map = {
        # "complex512": np.complex512,
        "complex256": np.complex256,
        "complex128": np.complex128,
        "complex64": np.complex64,
        # "float256": np.float256,
        "float128": np.float128,
        "float64": np.float64,
        "float32": np.float32,
    }

    dtype_str = os.environ.get("QSDTYPE", "").lower()
    if not dtype_str:
        return np.complex128  # default

    try:
        return dtype_map[dtype_str]
    except KeyError:
        raise ValueError(f"Unknown QSDTYPE value: {os.environ['QSDTYPE']}")
    
def set_dtype(dtype:str):
    assert type(dtype) == str, f"`dtype` should be a string but you provided {type(dtype)}."
    os.environ["QSDTYPE"] = dtype.lower()

