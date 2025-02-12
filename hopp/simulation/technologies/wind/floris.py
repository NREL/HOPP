# tools to add floris to the hybrid simulation class
from attrs import define, field
from dataclasses import dataclass, asdict
import csv
from typing import TYPE_CHECKING, Tuple
import numpy as np

from floris import FlorisModel, TimeSeries

from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
from hopp.type_dec import resource_file_converter
from pathlib import Path
from hopp.utilities import load_yaml
from hopp.tools.resource.wind_tools import (
    calculate_air_density_for_elevation, 
    parse_resource_data,
)
# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.wind.wind_plant import WindConfig


@define
class Floris(BaseClass):
    site: SiteInfo = field()
    config: "WindConfig" = field()
    verbose: bool = field(default = True)

    _operational_losses: float = field(init=False)
    _timestep: Tuple[int, int] = field(init=False)
    annual_energy_pre_curtailment_ac: float = field(init=False)
    fi: FlorisModel = field(init=False)

    def __attrs_post_init__(self):
        # 1) check that floris config is provided
        if self.config.floris_config is None:
            raise ValueError("A floris configuration must be provided")
        if self.config.timestep is None:
            raise ValueError("A timestep is required.")

        # 2) load floris config if needed
        if isinstance(self.config.floris_config,(str, Path)):
            # floris_config = Core.from_file(self.config.floris_config)
            floris_config = load_yaml(self.config.floris_config)
        else:
            floris_config = self.config.floris_config
        # the above change is a temporary patch to bridge to refactor floris

        # 3) modify air density in floris config if needed
        if self.config.adjust_air_density_for_elevation and self.site.elev is not None:
            rho = calculate_air_density_for_elevation(self.site.elev)
            floris_config["flow_field"].update({"air_density":rho})
        
        #initialize floris model
        self.fi = FlorisModel(floris_config)
        self._timestep = self.config.timestep
        self._operational_losses = self.config.operational_losses

        self.wind_resource_data = self.site.wind_resource.data #isn't this unnecessary?
        self.speeds, self.wind_dirs = parse_resource_data(self.site.wind_resource)

        self.wind_farm_xCoordinates = self.fi.layout_x
        self.wind_farm_yCoordinates = self.fi.layout_y
        self.nTurbs = len(self.wind_farm_xCoordinates)
        self.turb_rating = self.config.turbine_rating_kw
        
        self.wind_turbine_rotor_diameter = self.fi.core.farm.rotor_diameters[0]
        self.system_capacity = self.nTurbs * self.turb_rating

        # turbine power curve (array of kW power outputs)
        self.wind_turbine_powercurve_powerout = []

        # time to simulate
        if len(self.config.timestep) > 0:
            self.start_idx = self.config.timestep[0]
            self.end_idx = self.config.timestep[1]
        else:
            self.start_idx = 0
            self.end_idx = 8759

        # results
        self.gen = []
        self.annual_energy = None
        self.capacity_factor = None

        self.initialize_from_floris()

    def initialize_from_floris(self):
        """
        Please populate all the wind farm parameters
        """
        self.nTurbs = len(self.fi.layout_x) #this is redundant
        self.wind_turbine_powercurve_powerout = [1] * 30    # dummy for now

    def value(self, name: str, set_value=None):
        """
        if set_value = None, then retrieve value; otherwise overwrite variable's value
        """
        if set_value is not None:
            self.__setattr__(name, set_value)
        else:
            return self.__getattribute__(name)

    def set_floris_value(self,name,value):
        self.fi.set(**{name:value})

    def execute(self, project_life):
        
        if self.verbose:
            print('Simulating wind farm output in FLORIS...')

        # find generation of wind farm
        power_turbines = np.zeros((self.nTurbs, 8760))
        power_farm = np.zeros(8760)

        time_series = TimeSeries(
            wind_directions=self.wind_dirs[self.start_idx:self.end_idx],
            wind_speeds=self.speeds[self.start_idx:self.end_idx],
            turbulence_intensities=self.fi.core.flow_field.turbulence_intensities[0]
        )

        self.fi.set(wind_data=time_series)
        self.fi.run()

        power_turbines[:, self.start_idx:self.end_idx] = self.fi.get_turbine_powers().reshape((self.nTurbs, self.end_idx - self.start_idx))
        power_farm[self.start_idx:self.end_idx] = self.fi.get_farm_power().reshape((self.end_idx - self.start_idx))

        operational_efficiency = ((100 - self._operational_losses)/100)
        # Adding losses from PySAM defaults (excluding turbine and wake losses)
        self.gen = power_farm * operational_efficiency / 1000 # kW

        self.annual_energy = np.sum(self.gen) # kWh
        self.capacity_factor = np.sum(self.gen) / (8760 * self.system_capacity) * 100
        self.turb_powers = power_turbines * operational_efficiency / 1000 # kW
        self.turb_velocities = self.fi.turbine_average_velocities
        self.annual_energy_pre_curtailment_ac = np.sum(self.gen) # kWh

    def export(self):
        """
        Return all the floris system configuration in a dictionary for the financial model
        """
        config = {
            'system_capacity': self.system_capacity,
            'annual_energy': self.annual_energy,
        }
        return config