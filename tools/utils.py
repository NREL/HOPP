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

def export_all(obj):
    """
    Exports all variables from pysam objects including those not assigned

    Assumes the object is a collection of objects with all the variables within them:
    obj:
        object1:
            variable1:
            variable2:
        <no variables here not contained in objects>
    """
    output_dict = {}
    for attribute_name in dir(obj):
        try:
            attribute = getattr(obj, attribute_name)
        except:
            continue
        if not callable(attribute) and not attribute_name.startswith('__'):
            output_dict[attribute_name] = {}
            for subattribute_name in dir(attribute):
                if subattribute_name.startswith('__'):
                    continue
                try:
                    subattribute = getattr(attribute, subattribute_name)
                except Exception as e:
                    if 'not assigned' in str(e):
                        output_dict[attribute_name][subattribute_name] = None
                        continue
                    else:
                        continue
                if not callable(subattribute):
                    output_dict[attribute_name][subattribute_name] = subattribute

            # Remove dictionary if empty
            if len(output_dict[attribute_name]) == 0:
                del output_dict[attribute_name]

    return output_dict

def array_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return isinstance(array, Sequence) or (isinstance(array, np.ndarray) and hasattr(array, "__len__"))