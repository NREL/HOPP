from attrs import define, field

import PySAM.Windpower as Windpower

from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
from hopp.type_dec import resource_file_converter
from hopp.utilities.utilities import yml2dict


@define
class PySAMWind(BaseClass):
    config_dict: dict = field(converter=dict)
    site: SiteInfo = field()
    timestep: tuple = field(default=(), converter=tuple)

    system_model: Windpower = field(init=False)

    def __attrs_post_init__(self):
        input_file_path = resource_file_converter(self.config_dict["simulation_input_file"])
        input_dict = yml2dict(input_file_path)

        self.system_model = Windpower.new()
        self.system_model.assign(input_dict)

        self.wind_turbine_rotor_diameter = input_dict['Turbine']['wind_turbine_rotor_diameter']
        self.wind_farm_xCoordinates = input_dict['Farm']['wind_farm_xCoordinates']
        self.wind_farm_yCoordinates = input_dict['Farm']['wind_farm_yCoordinates']
        self.nTurbs = len(self.wind_farm_xCoordinates)
        self.system_model.value("wind_resource_data", self.site.wind_resource.data)

        # turbine power curve (array of kW power outputs)
        self.wind_turbine_powercurve_powerout = [1] * self.nTurbs

    def value(self, name: str, set_value=None):
        """
        if set_value = None, then retrieve value; otherwise overwrite variable's value
        """
        if set_value:
            self.__setattr__(name, set_value)
        else:
            return self.__getattribute__(name)

    def execute(self, project_life):
        self.system_model.execute(project_life)

    @property
    def annual_energy(self):
        return self.system_model.value("annual_energy")

    @annual_energy.setter
    def annual_energy(self, d):
        self.system_model.value("annual_energy", d)

    @property
    def gen(self):
        return self.system_model.value("gen")

    @gen.setter
    def gen(self, d):
        self.system_model.value("gen", d)
