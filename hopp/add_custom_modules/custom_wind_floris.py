# tools to add floris to the hybrid simulation class
import numpy as np
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface
import csv
import yaml
import os


class Floris:

    def __init__(self, config_dict, site, timestep=()):

        self.fi = FlorisInterface(config_dict["floris_config"])

        self.site = site
        self.wind_resource_data = self.site.wind_resource.data
        self.speeds, self.wind_dirs = self.parse_resource_data()

        save_data = np.zeros((len(self.speeds),2))
        save_data[:,0] = self.speeds
        save_data[:,1] = self.wind_dirs

        data_path = 'speed_dir_data.csv'
        if not os.path.exists(data_path):
            with open(data_path, 'w', newline='') as fo:
                writer = csv.writer(fo)
                writer.writerows(save_data)

        self.wind_farm_xCoordinates = self.fi.layout_x
        self.wind_farm_yCoordinates = self.fi.layout_y
        self.nTurbs = len(self.wind_farm_xCoordinates)
        self.turb_rating = config_dict["turbine_rating_kw"]
        self.wind_turbine_rotor_diameter = self.fi.floris.farm.rotor_diameters[0]
        self.system_capacity = self.nTurbs * self.turb_rating

        # turbine power curve (array of kW power outputs)
        self.wind_turbine_powercurve_powerout = []

        # time to simulate
        if len(timestep) > 0:
            self.start_idx = timestep[0]
            self.end_idx = timestep[1]
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

    def execute(self, project_life, verbose=False):

        if verbose:
            print('Simulating wind farm output in FLORIS...')

        # find generation of wind farm
        power_turbines = np.zeros((self.nTurbs, 8760))
        power_farm = np.zeros(8760)

        self.fi.reinitialize(wind_speeds=self.speeds[self.start_idx:self.end_idx], wind_directions=self.wind_dirs[self.start_idx:self.end_idx], time_series=True)
        self.fi.calculate_wake()

        power_turbines[:, self.start_idx:self.end_idx] = self.fi.get_turbine_powers().reshape((self.nTurbs, self.end_idx - self.start_idx))
        power_farm[self.start_idx:self.end_idx] = self.fi.get_farm_power().reshape((self.end_idx - self.start_idx))

        # Adding losses from PySAM defaults (excluding turbine and wake losses)
        self.gen = power_farm *((100 - 12.83)/100) / 1000
        # self.gen = power_farm  / 1000
        self.annual_energy = np.sum(self.gen)
        if verbose:
            print('Wind annual energy: ', self.annual_energy)
        self.capacity_factor = np.sum(self.gen) / (8760 * self.system_capacity) * 100
