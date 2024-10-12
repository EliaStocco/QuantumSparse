# some functions ...
import numpy as np
import bisect
import functools
from typing import List

MAXSIZE = None

def prepare_opts(opts):
    opts = {} if opts is None else opts
    opts["print"]       = None   if "print"       not in opts else opts["print"]
    opts["sort"]        = True   if "sort"       not in opts else opts["sort"]
    opts["check-low-T"] = 0  if "check-low-T" not in opts else opts["check-low-T"]
    opts["inplace"]     = True   if "inplace"       not in opts else opts["inplace"]
    #opts["return-S"] = False if "return-S" not in opts else opts["return-S"]
    return opts

#
# def magnetic_moment_operator(Sx,Sy,Sz,opts=None):
#     Mx,My,Mz = 0,0,0
#     for sx,sy,sz in zip(Sx,Sy,Sz):
#         Mx += g*muB*sx
#         My += g*muB*sy
#         Mz += g*muB*sz    
#     return Mx,My,Mz

def spherical_coordinates(r,theta,phi,cos=np.cos,sin=np.sin):
    x = r*cos(phi)*sin(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(theta)
    return x,y,z

# @functools.lru_cache(maxsize=MAXSIZE)
def first_larger_than_N(sorted_list:List[int], N:int):
    """
    Returns the first element in a sorted list that is larger than the given number N.
    
    Parameters:
    - sorted_list (list): A sorted list of elements.
    - N (int): The number to compare with the elements in the list.
    
    Returns:
    - int or None: The index of the first element larger than N if found, otherwise None.
    """
    # Find the index where N would go to keep the list sorted
    index = bisect.bisect_right(sorted_list, N)
    
    # If the index is within the bounds of the list, return the element at that index
    if index < len(sorted_list):
        return index
    else:
        return None  # If no element is larger than N