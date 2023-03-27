import numpy as np
from typing import Sequence


def flatten_dict(d):
    def get_key_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                yield from get_key_values(value)
            else:
                yield key, value

    return {key:value for (key,value) in get_key_values(d)}

def array_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return isinstance(array, Sequence) or (isinstance(array, np.ndarray) and hasattr(array, "__len__"))