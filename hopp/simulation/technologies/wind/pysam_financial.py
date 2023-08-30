# tools to add floris to the hybrid simulation class
from attrs import define, field

import PySAM.Singleowner as Singleowner

from hopp.simulation.base import BaseClass
from hopp.type_dec import resource_file_converter
from hopp.utilities.utilities import load_yaml


@define
class PySAMFinancial(BaseClass):
    config_dict: dict = field(converter=dict)

    financial_model: Singleowner = field(init=False)

    def __attrs_post_init__(self):
        input_file_path = resource_file_converter(self.config_dict["financial_input_file"])
        filetype = input_file_path.suffix.strip(".")
        if filetype.lower() in ("yml", "yaml"):
            input_dict = load_yaml(input_file_path)
        else:
            raise ValueError("Supported import filetype is YAML")
            
        self.financial_model = Singleowner.new()
        self.financial_model.assign(input_dict)
