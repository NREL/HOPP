# tools to add floris to the hybrid simulation class
import numpy as np
import matplotlib.pyplot as plt
import floris.tools as wfct


class Floris:

    def __init__(self, config_dict, site, timestep=()):

        self.fi = wfct.floris_interface.FlorisInterface(input_dict=config_dict["floris_config"])

        self.site = site
        self.wind_resource_data = self.site.wind_resource.data
        self.speeds, self.wind_dirs = self.parse_resource_data()

        self.wind_farm_xCoordinates = self.fi.layout_x
        self.wind_farm_yCoordinates = self.fi.layout_y
        self.nTurbs = len(self.wind_farm_xCoordinates)
        self.turb_rating = config_dict["turbine_rating_kw"]
        self.wind_turbine_rotor_diameter = self.fi.floris.farm.turbines[0].rotor_diameter
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

        for i in range(self.start_idx, self.end_idx):

            # reinitialize floris with the wind speed and wind direction
            self.fi.reinitialize_flow_field(wind_speed=self.speeds[i], wind_direction=self.wind_dirs[i])

            # Calculate wake
            self.fi.calculate_wake()

            # outputs
            power_turbines[:, i] = self.fi.get_turbine_power()
            power_farm[i] = self.fi.get_farm_power()

        self.gen = power_farm / 1000
        self.annual_energy = np.sum(self.gen)
        print('Wind annual energy: ', self.annual_energy)
        self.capacity_factor = np.sum(self.gen) / (8760 * self.system_capacity)







