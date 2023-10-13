import os
import yaml


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)


Loader.add_constructor('!include', Loader.include)

def load_yaml(filename, loader=Loader):
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with open(filename) as fid:
        return yaml.load(fid, loader)

def yml2dict(input_file_path):
    filetype = input_file_path.suffix.strip(".")
    if filetype.lower() in ("yml", "yaml"):
        input_dict = load_yaml(input_file_path)
        return input_dict
    else:
        raise ValueError("Supported import filetype is YAML")
