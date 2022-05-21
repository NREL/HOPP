# tools to add floris to the hybrid simulation class
import numpy as np
import matplotlib.pyplot as plt
import floris

floris_version = float(floris.__version__[0:2])
if floris_version >= 3.1:
    from floris.tools import FlorisInterface

class Floris:

    def __init__(self, config_dict, site, timestep=()):

        if floris_version < 3.0:
            raise EnvironmentError("Floris v3.1 or higher is required")

        self.fi = FlorisInterface(config_dict["floris_config"])

        self.site = site
        self.wind_resource_data = self.site.wind_resource.data
        self.speeds, self.wind_dirs = self.parse_resource_data()

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
        for i in range((len(self.site.wind_resource.data['data']))):
            speeds[i] = self.site.wind_resource.data['data'][i][2]
            wind_dirs[i] = self.site.wind_resource.data['data'][i][3]

        return speeds, wind_dirs

    def execute(self, project_life):

        print('Simulating wind farm output in FLORIS...')

        # find generation of wind farm
        power_turbines = np.zeros((self.nTurbs, 8760))
        power_farm = np.zeros(8760)

        self.fi.reinitialize(wind_speeds=self.speeds[self.start_idx:self.end_idx], wind_directions=self.wind_dirs[self.start_idx:self.end_idx])
        self.fi.calculate_wake()

        powers = self.fi.get_turbine_powers()
        power_turbines[:, self.start_idx:self.end_idx] = powers[0].reshape((self.nTurbs, self.end_idx - self.start_idx))

        power_farm = np.array(power_turbines).sum(axis=0)

        self.gen = power_farm / 1000
        self.annual_energy = np.sum(self.gen)
        print('Wind annual energy: ', self.annual_energy)
        self.capacity_factor = np.sum(self.gen) / (8760 * self.system_capacity)
