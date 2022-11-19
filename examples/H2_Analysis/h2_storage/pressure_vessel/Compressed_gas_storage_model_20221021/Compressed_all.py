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
from .Compressed_gas_function import CompressedGasFunction

class PressureVessel():
    def __init__(self, Wind_avai=80, H2_flow=200, cdratio=1, Energy_cost=0.07, cycle_number=1, spread_sheet_path="./Tankinator.xlsx"):

        ########Key inputs##########
        self.Wind_avai = Wind_avai  #Wind availability in %
        self.H2_flow = H2_flow  #Flow rate of steel plants in tonne/day
        self.cdratio = cdratio  #Charge/discharge ratio, for example 2 means the charging is 2x faster than discharge
        self.Energy_cost = Energy_cost  #Renewable energy cost in $/kWh


        #######Other inputs########
        self.cycle_number = cycle_number #Equivalent cycle number for a year, only affects operation (the higher the number is the less effect there will be), set as now as I am not sure how the maximum sotrage capacity is determined and how the storage will be cycled
        self.path_tankinator= spread_sheet_path

        self.compressed_gas_function = CompressedGasFunction

    def run(self):
        #####Run calculation########
        self.compressed_gas_function.func(self.Wind_avai, self.H2_flow, self.cdratio, self.Energy_cost, self.cycle_number, self.path_tankinator)

        ########Outputs################

        ######Maximum equivalent storage capacity and duration
        self.capacity_max = self.compressed_gas_function.capacity_max   #This is the maximum equivalent H2 storage in kg
        self.t_discharge_hr_max = self.compressed_gas_function.t_discharge_hr_max   #This is tha maximum storage duration in kg

        ###Parameters for capital cost fitting for optmizing capital cost
        self.a_fit_capex = self.compressed_gas_function.a_fit
        self.b_fit_capex = self.compressed_gas_function.b_fit

        #Parameters for operational cost fitting for optmizing capital cost
        self.a_fit_opex = self.compressed_gas_function.a_op_fit
        self.b_fit_opex = self.compressed_gas_function.b_op_fit
        self.c_fit_opex = self.compressed_gas_function.c_op_fit


if __name__ == "__main__":
    storage = PressureVessel()
    storage.run()


