# tools to add floris to the hybrid simulation class
from attrs import define, field
import csv
from typing import TYPE_CHECKING, Tuple
import numpy as np

from floris.tools import FlorisInterface

from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
from hopp.type_dec import resource_file_converter

# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.wind.wind_plant import WindConfig


@define
class Floris(BaseClass):
    site: SiteInfo = field()
    config: "WindConfig" = field()

    _timestep: Tuple[int, int] = field(init=False)
    fi: FlorisInterface = field(init=False)
    _operational_losses: float = field(init=False)

    def __attrs_post_init__(self):
        # floris_input_file = resource_file_converter(self.config["simulation_input_file"])
        floris_input_file = self.config.floris_config # DEBUG!!!!!

        if floris_input_file is None:
            raise ValueError("A floris configuration must be provided")
        if self.config.timestep is None:
            raise ValueError("A timestep is required.")

        # the above change is a temporary patch to bridge to refactor floris

        self.fi = FlorisInterface(floris_input_file)
        self._timestep = self.config.timestep
        self._operational_losses = self.config.operational_losses

        self.wind_resource_data = self.site.wind_resource.data
        self.speeds, self.wind_dirs = self.parse_resource_data()

        save_data = np.zeros((len(self.speeds),2))
        save_data[:,0] = self.speeds
        save_data[:,1] = self.wind_dirs

        with open('speed_dir_data.csv', 'w', newline='') as fo:
            writer = csv.writer(fo)
            writer.writerows(save_data)

        self.wind_farm_xCoordinates = self.fi.layout_x
        self.wind_farm_yCoordinates = self.fi.layout_y
        self.nTurbs = len(self.wind_farm_xCoordinates)
        self.turb_rating = self.config.turbine_rating_kw
        self.wind_turbine_rotor_diameter = self.fi.floris.farm.rotor_diameters[0]
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
        self.nTurbs = len(self.fi.layout_x)
        self.wind_turbine_powercurve_powerout = [1] * 30    # dummy for now
        pass

    def value(self, name: str, set_value=None):
        """
        if set_value = None, then retrieve value; otherwise overwrite variable's value
        """
        if set_value:
            self.__setattr__(name, set_value)
        else:
            return self.__getattribute__(name)

    def parse_resource_data(self):

        # extract data for simulation
        speeds = np.zeros(len(self.wind_resource_data['data']))
        wind_dirs = np.zeros(len(self.site.wind_resource.data['data']))
        data_rows_total = 4
        if np.shape(self.site.wind_resource.data['data'])[1] > data_rows_total:
            height_entries = int(np.round(np.shape(self.site.wind_resource.data['data'])[1]/data_rows_total))
            data_entries = np.empty((height_entries))
            for j in range(height_entries):
                data_entries[j] = int(j*data_rows_total)
            data_entries = data_entries.astype(int)
            for i in range((len(self.site.wind_resource.data['data']))):
                data_array = np.array(self.site.wind_resource.data['data'][i])
                speeds[i] = np.mean(data_array[2+data_entries])
                wind_dirs[i] = np.mean(data_array[3+data_entries])
        else:
            for i in range((len(self.site.wind_resource.data['data']))):
                speeds[i] = self.site.wind_resource.data['data'][i][2]
                wind_dirs[i] = self.site.wind_resource.data['data'][i][3]

        return speeds, wind_dirs

    def execute(self, project_life):

        print('Simulating wind farm output in FLORIS...')

        # find generation of wind farm
        power_turbines = np.zeros((self.nTurbs, 8760))
        power_farm = np.zeros(8760)

        self.fi.reinitialize(wind_speeds=self.speeds[self.start_idx:self.end_idx], wind_directions=self.wind_dirs[self.start_idx:self.end_idx], time_series=True)
        self.fi.calculate_wake()

        power_turbines[:, self.start_idx:self.end_idx] = self.fi.get_turbine_powers().reshape((self.nTurbs, self.end_idx - self.start_idx))
        power_farm[self.start_idx:self.end_idx] = self.fi.get_farm_power().reshape((self.end_idx - self.start_idx))

        # Adding losses from PySAM defaults (excluding turbine and wake losses)
        self.gen = power_farm * ((100 - self._operational_losses)/100) / 1000 # kW

        self.annual_energy = np.sum(self.gen) # kWh
        self.capacity_factor = np.sum(self.gen) / (8760 * self.system_capacity) * 100

        self.turb_powers = power_turbines * (100 - self._operational_losses) / 100 / 1000 # kW
        self.turb_velocities = self.fi.turbine_average_velocities