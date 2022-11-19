# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:08:09 2022
@author: ppeng

Revisions:
- 20221118:
    Author: Jared J. Thomas
    Description: 
        - Reformatted to be a class
"""
# package imports
import os

# local imports
from .compressed_gas_function import CompressedGasFunction

class PressureVessel():
    def __init__(self, Wind_avai=80, H2_flow=200, cdratio=1, Energy_cost=0.07, cycle_number=1, parent_path=os.path.abspath(os.path.dirname(__file__)), spread_sheet_name="Tankinator.xlsx"):

        ########Key inputs##########
        self.Wind_avai = Wind_avai  #Wind availability in %
        self.H2_flow = H2_flow  #Flow rate of steel plants in tonne/day
        self.cdratio = cdratio  #Charge/discharge ratio, for example 2 means the charging is 2x faster than discharge
        self.Energy_cost = Energy_cost  #Renewable energy cost in $/kWh

        #######Other inputs########
        self.cycle_number = cycle_number #Equivalent cycle number for a year, only affects operation (the higher the number is the less effect there will be), set as now as I am not sure how the maximum sotrage capacity is determined and how the storage will be cycled

        self.compressed_gas_function = CompressedGasFunction(path_tankinator=os.path.join(parent_path, spread_sheet_name))

    def run(self):
        #####Run calculation########
        self.compressed_gas_function.func(Wind_avai=self.Wind_avai, H2_flow=self.H2_flow, cdratio=self.cdratio, Energy_cost=self.Energy_cost, cycle_number=self.cycle_number)

        ########Outputs################

        ######Maximum equivalent storage capacity and duration
        self.capacity_max = self.compressed_gas_function.capacity_max   #This is the maximum equivalent H2 storage in kg
        self.t_discharge_hr_max = self.compressed_gas_function.t_discharge_hr_max   #This is tha maximum storage duration in kg

        ###Parameters for capital cost fitting for optimizing capital cost
        self.a_fit_capex = self.compressed_gas_function.a_cap_fit
        self.b_fit_capex = self.compressed_gas_function.b_cap_fit

        #Parameters for operational cost fitting for optimizing capital cost
        self.a_fit_opex = self.compressed_gas_function.a_op_fit
        self.b_fit_opex = self.compressed_gas_function.b_op_fit
        self.c_fit_opex = self.compressed_gas_function.c_op_fit

    def plot(self):
        self.compressed_gas_function.plot()

if __name__ == "__main__":
    storage = PressureVessel()
    storage.run()


