# implement a simple dispatch model

import numpy as np

class SimpleDispatch():

    def __init__(self):

        # length of simulation
        self.Nt = 1
        
        # amount of curtailment experienced by plant (MW)
        self.curtailment = np.zeros(self.Nt)        

        # amount of energy needed from the battery (MW)
        self.shortfall = np.zeros(self.Nt)          
        
        # size of battery (MWh)
        self.battery_storage = 0

        # Charge rate of the battery (MW)
        self.charge_rate = 0
        self.discharge_rate = 0

        # Battery SOC limits and initial state (range 0-1)
        self.initial_SOC = 0
        self.max_SOC = 1
        self.min_SOC = 0
        

    def run(self):

        # storage module
        battery_storage = self.battery_storage  # MWh
        charge_rate = self.charge_rate  # MW
        discharge_rate = self.discharge_rate  #MW

        battery_SOC = np.zeros(self.Nt)
        battery_used = np.zeros(self.Nt)
        excess_energy = np.zeros(self.Nt)

        battery_SOC[0] = self.initial_SOC * self.battery_storage
        battery_max_SOC = self.max_SOC * self.battery_storage
        battery_min_SOC = self.min_SOC * self.battery_storage

        for i in range(self.Nt):
            # should you charge
            if self.curtailment[i] > 0:
                if i == 0:
                    battery_SOC[i] = np.max(self.initial_SOC + \
                        np.min([self.curtailment[i], charge_rate]),battery_max_SOC)
                    amount_charged = battery_SOC[i] - self.initial_SOC
                    excess_energy[i] = self.curtailment[i] - amount_charged
                else:
                    if battery_SOC[i-1] < battery_max_SOC:
                        add_gen = np.min([self.curtailment[i], charge_rate])
                        battery_SOC[i] = np.min([battery_SOC[i-1] + add_gen, battery_max_SOC])
                        amount_charged = battery_SOC[i] - battery_SOC[i-1]
                        excess_energy[i] = self.curtailment[i] - amount_charged
                    else:
                        battery_SOC[i] = battery_SOC[i-1]
                        excess_energy[i] = self.curtailment[i]

            # should you discharge
            else:
                if i > 0:
                    if battery_SOC[i-1] > battery_min_SOC:
                        
                        battery_used[i] = np.min([self.shortfall[i], battery_SOC[i-1],discharge_rate])
                        battery_SOC[i] = battery_SOC[i-1] - battery_used[i]
        # print('==============================================')
        # print('Battery Generation: ', np.sum(battery_used))
        # print('Amount of energy going to the grid: ', np.sum(excess_energy))
        # print('==============================================')
        return battery_used, excess_energy, battery_SOC



