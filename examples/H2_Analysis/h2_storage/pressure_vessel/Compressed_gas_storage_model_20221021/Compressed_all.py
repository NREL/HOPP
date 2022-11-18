# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:08:09 2022

@author: ppeng
"""
import Compressed_gas_function 

class pressure_vessel():
    def __init__(self):
        self.Wind_avai = 8
        ########Key inputs##########

        self.Wind_avai =80  #Wind availability in %
        self.H2_flow = 200  #Flow rate of steel plants in tonne/day
        self.cdratio = 1  #Charge/discharge ratio, for example 2 means the charging is 2x faster than discharge
        self.Energy_cost = 0.07  #Renewable energy cost in $/kWh


        #######Other inputs########
        self.cycle_number = 1 #Equivalent cycle number for a year, only affects operation (the higher the number is the less effect there will be), set as now as I am not sure how the maximum sotrage capacity is determined and how the storage will be cycled
        self.path_tankinator= "./Tankinator.xlsx"

    def main(self):
        #####Run calculation########
        Compressed_gas_function.func(self.Wind_avai, self.H2_flow, self.cdratio, self.Energy_cost, self.cycle_number, self.path_tankinator)

        ########Outputs################

        ######Maximum equivalent storage capacity and duration
        Compressed_gas_function.capacity_max   #This is the maximum equivalent H2 storage in kg
        Compressed_gas_function.t_discharge_hr_max   #This is tha maximum storage duration in kg

        ###Parameters for capital cost fitting for optmizing capital cost
        self.a_fit = Compressed_gas_function.a_fit
        self.b_fit = Compressed_gas_function.b_fit

        #Parameters for operational cost fitting for optmizing capital cost
        self.a_op_fit = Compressed_gas_function.a_op_fit
        self.b_op_fit = Compressed_gas_function.b_op_fit
        self.c_op_fit = Compressed_gas_function.c_op_fit


if __name__ == "__main__":
    storage = pressure_vessel()
    storage.main()


