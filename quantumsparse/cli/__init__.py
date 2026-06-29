from typing import Union
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
    
def size_type(s: str, dtype=float, N=None):
    s = s.replace("[", "").replace("]", "")
    if "," in s:
        s = s.replace(",", " ")
    s = s.split()
    if N is not None and len(s) != N:
        raise ValueError(f"You should provide {N} values")
    values = []
    for k in s:
        if k.lower() == "none":
            values.append(None)
        else:
            values.append(dtype(k))
    return values  # return list, not np.array, so None stays


def flist(s):
    return size_type(s, float)  # float list


def ilist(s):
    return size_type(s, int)  # integer list


def slist(s):
    return size_type(s, str)  # string list

#------------------#
def is_convertible_to_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#------------------#
def string2index(stridx: str) -> Union[int, slice, str]:
    """Convert index string to either int or slice"""
    if ':' not in stridx:
        # may contain database accessor
        try:
            return int(stridx)
        except ValueError:
            return stridx
    i = [None if s == '' else int(s) for s in stridx.split(':')]
    return slice(*i)

def str2index(index):
    """
    Convert integer index to slice string.

    Args:
        index: Index to convert.

    Returns:
        slice: Converted slice.
    """
    if isinstance(index, slice):
        return index
    
    if is_convertible_to_integer(index):
        index=int(index)

    if isinstance(index, int):
        return string2index(f"{index}:{index+1}")
    elif index is None:
        return slice(None,None,None)
    elif isinstance(index, str):
        try:
            return string2index(index)
        except:
            raise ValueError("error creating slice from string {:s}".format(index))
    # elif isinstance(index, slice):
    #     return index
    else:
        raise ValueError("`index` can be int, str, or slice, not {}".format(index))