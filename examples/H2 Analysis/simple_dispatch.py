# implement a simple dispatch model

import numpy as np

class SimpleDispatch():

    def __init__(self, combined_pv_wind_curtailment_hopp, energy_shortfall_hopp, N, size_battery):

        # amount of curtailment experienced by plant
        self.curtailment = combined_pv_wind_curtailment_hopp

        # amount of energy needed from the battery
        self.shortfall = energy_shortfall_hopp

        # length of simulation
        self.Nt = N

        # size of battery (assumed to be the same as the charge rate per hour)
        self.size_battery = size_battery


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
                    battery_SOC = np.min([self.curtailment[i], charge_rate])
                    print(battery_SOC)
                else:
                    battery_SOC[i] = battery_SOC[i - 1]
                if battery_SOC[i-1] < rated_size:
                    add_gen = np.min([self.curtailment[i], charge_rate])
                    battery_SOC[i] = np.min([battery_SOC[i] + add_gen, rated_size])
            else:
                battery_SOC[i] = battery_SOC[i - 1]

            # should you discharge
            if self.shortfall[i] > 0 and battery_SOC[i] > 0:
                energy_used = np.min([self.shortfall[i], battery_SOC[i]])
                battery_SOC[i] = battery_SOC[i] - energy_used
                battery_used[i] = energy_used

            # overall the amount of energy you could have sold to the grid assuming perfect knowledge
            if battery_SOC[i] == rated_size and battery_SOC[i-1] == rated_size:
                excess_energy[i] = self.curtailment[i]
            else:
                excess_energy[i] = np.max([self.curtailment[i] - battery_SOC[i], 0.0])

        print('==============================================')
        print('Battery Generation: ', np.sum(battery_used))
        print('Amount of energy going to the grid: ', np.sum(excess_energy))
        print('==============================================')

        return battery_used, excess_energy, battery_SOC



