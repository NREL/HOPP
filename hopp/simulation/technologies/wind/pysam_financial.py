# tools to add floris to the hybrid simulation class
from attrs import define, field

import PySAM.Singleowner as Singleowner

from hopp.simulation.base import BaseClass
from hopp.type_dec import resource_file_converter
from hopp.utilities.utilities import yml2dict


@define
class PySAMFinancial(BaseClass):
    config_dict: dict = field(converter=dict)

    financial_model: Singleowner = field(init=False)

    def __attrs_post_init__(self):
        input_file_path = resource_file_converter(self.config_dict["financial_input_file"])
        input_dict = yml2dict(input_file_path)
            
        self.financial_model = Singleowner.new()
        self.financial_model.assign(input_dict)
