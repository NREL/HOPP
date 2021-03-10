# tools to add floris to the hybrid simulation class

import numpy as np
import matplotlib.pyplot as plt


class Floris():

    def __init__(self, fi, site, rated_power):

        self.fi = fi
        self.site = site
        self.speeds, self.wind_dirs = self.parse_resource_data()
        self.nTurbs = len(self.fi.get_turbine_power())
        self.rated_power = rated_power

    def parse_resource_data(self):

        # extract data for simulation
        speeds = np.zeros(len(self.site.wind_resource.data['data']))
        wind_dirs = np.zeros(len(self.site.wind_resource.data['data']))
        for i in range((len(self.site.wind_resource.data['data']))):
            speeds[i] = self.site.wind_resource.data['data'][i][2]
            wind_dirs[i] = self.site.wind_resource.data['data'][i][3]

        return speeds, wind_dirs

    def simulate(self):

        print('Simulating wind farm output in FLORIS...')

        # find generation of wind farm
        power_turbines = np.zeros((self.nTurbs,self.site.n_timesteps))
        power_farm = np.zeros(self.site.n_timesteps)
        for i in range(self.site.n_timesteps):

            print('Calculating time step', i, 'out of ', self.site.n_timesteps)

            # reinitialize floris with the wind speed and wind direction
            self.fi.reinitialize_flow_field(wind_speed=self.speeds[i], wind_direction=self.wind_dirs[i])

            # Calculate wake
            self.fi.calculate_wake()

            # outputs
            power_turbines[:,i] = self.fi.get_turbine_power()
            power_farm[i] = self.fi.get_farm_power()

        self.generation_profile = power_farm

    @property
    def annual_energy_kw(self):
        # compute the annual energy production in kW

        # TODO: implement wind rose
        return np.sum(self.generation_profile) / 1000

    @property
    def capacity_factors(self):
        # compute the capacity factor

        return self.annual_energy_kw / (self.rated_power * 8760)

