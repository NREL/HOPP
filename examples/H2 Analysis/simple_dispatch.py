# implement a simple dispatch model

import numpy as np

class SimpleDispatch():

    def __init__(self, combined_pv_wind_curtailment_hopp, energy_shortfall_hopp, N, size_battery):

        # amount of curtailment experienced by plant
        self.curtailment = combined_pv_wind_curtailment_hopp

        # amount of energy needed from the battery
        self.shortfall = energy_shortfall_hopp
        # print("Energy shortfall in battery simulation is: {}".format(energy_shortfall_hopp))

        # length of simulation
        self.Nt = N
        # print("Length of battery simulation is: {}".format(N))

        # size of battery (assumed to be the same as the charge rate per hour)
        self.size_battery = size_battery
        # print("Size Battery in battery simulation is: {}".format(size_battery))


    def run(self):

        # storage module
        rated_size = self.size_battery # kW -> 200 MW
        charge_rate = self.size_battery # kWH -> 200 MWh
        battery_SOC = np.zeros(self.Nt)
        battery_used = np.zeros(self.Nt)
        excess_energy = np.zeros(self.Nt)
        
        for i in range(self.Nt):
            # should you charge
            if self.curtailment[i] > 0:
                if i == 0:
                    battery_SOC[i] = np.min([self.curtailment[i], charge_rate])
                    excess_energy[i] = self.curtailment[i] - np.min([self.curtailment[i], charge_rate])
                else:
                    if battery_SOC[i-1] < rated_size:
                        add_gen = np.min([self.curtailment[i], charge_rate])
                        battery_SOC[i] = np.min([battery_SOC[i-1] + add_gen, rated_size])
                        excess_energy[i] = self.curtailment[i] - add_gen
                    else:
                        battery_SOC[i] = battery_SOC[i - 1]
                        excess_energy[i] = self.curtailment[i]

            # should you discharge
            else:
                if i > 0:
                    battery_SOC[i] = battery_SOC[i-1]
                    if battery_SOC[i] > 0:
                        
                        energy_used = np.min([self.shortfall[i], battery_SOC[i]])
                        battery_SOC[i] = battery_SOC[i] - energy_used
                        battery_used[i] = energy_used

        # print('==============================================')
        # print('Battery Generation: ', np.sum(battery_used))
        # print('Amount of energy going to the grid: ', np.sum(excess_energy))
        # print('==============================================')
        return battery_used, excess_energy, battery_SOC



