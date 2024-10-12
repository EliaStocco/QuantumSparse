# some functions ...
import numpy as np
import bisect
from typing import List
import sys
from collections import deque

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
    
def get_deep_size(obj, seen=None):
    """Recursively find the memory footprint of a Python object, including referenced objects."""
    if seen is None:
        seen = set()
        
    obj_id = id(obj)
    
    if obj_id in seen:
        return 0  # To avoid counting the same object multiple times
    
    # Mark the object as seen
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, deque)):
        size += sum(get_deep_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(vars(obj), seen)
    elif hasattr(obj, '__slots__'):
        size += sum(get_deep_size(getattr(obj, slot), seen) for slot in obj.__slots__ if hasattr(obj, slot))
    
    return size

def get_energy_levels(values,N=16):
    
    old_values = values.copy()
    
    deltaE = np.max(values) - np.min(values)
    values = (values - np.min(values))/deltaE
    
    assert np.min(values) == 0, "coding error"
    assert np.max(values) == 1, "coding error"
    
    # exponents = np.arange(1,N+1)
    # grid = np.power(base,exponents)
    
    Nlist = np.zeros(N,dtype=int)
    for n in range(N):
        tmp = np.round(values,n)
        Nlist[n] = len(np.unique(tmp))
        
    # number of energy levels
    Nel = np.median(Nlist)
    
    # choose the rounding
    n = np.where(Nlist==Nel)[0][0]
    
    tmp = np.round(values,n)
    w,ii,counts = np.unique(tmp,return_inverse=True,return_counts=True)
    
    assert w.shape == Nel, "coding error"
    assert np.allclose(w[ii],values,atol=np.power(0.1,n)), "coding error"
    
    u = np.unique(ii)
    assert u.shape == w.shape, "coding error"
    assert np.allclose(u,np.arange(len(w))), "coding error"
    
    
    w = w*deltaE + np.min(old_values)
    return w,counts,ii