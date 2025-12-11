import os
import yaml
import numpy as np
from typing import Sequence


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)


Loader.add_constructor('!include', Loader.include)

def load_yaml(filename, loader=Loader) -> dict:
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with open(filename) as fid:
        return yaml.load(fid, loader)

def check_create_folder(filepath):
    already_exists = True
    if not os.path.isdir(filepath):
        os.makedirs(filepath,exist_ok=True)
        already_exists = False
    return already_exists

def write_yaml(filename,data):
    if not '.yaml' in filename:
        filename = filename +'.yaml'

    with open(filename, 'w+') as file:
        yaml.dump(data, file,sort_keys=False,encoding = None,default_flow_style=False)


def flatten_dict(d):
    def get_key_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                yield from get_key_values(value)
            else:
                yield key, value

    return {key:value for (key,value) in get_key_values(d)}

def equal(a, b):
    """Determines whether integers, floats, lists, tuples or dictionaries are equal"""
    if isinstance(a, (int, float)):
        return np.isclose(a, b)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        else:
            for i in range(len(a)):
                if not np.isclose(a[i], b[i]):
                    return False
            return True
    elif isinstance(a, dict):
        if len(a) != len(b):
            return False
        else:
            for key in a.keys():
                if key not in b.keys():
                    return False
                if not np.isclose(a[key], b[key]):
                    return False
            return True
    else:
        raise Exception('Type not recognized')

def array_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return isinstance(array, Sequence) or (isinstance(array, np.ndarray) and hasattr(array, "__len__"))