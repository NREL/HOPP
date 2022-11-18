# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:08:09 2022

@author: ppeng
"""
import Compressed_gas_function 


########Key inputs##########

Wind_avai =80  #Wind availability in %
H2_flow = 200  #Flow rate of steel plants in tonne/day
cdratio = 1  #Charge/discharge ratio, for example 2 means the charging is 2x faster than discharge
Energy_cost = 0.07  #Renewable energy cost in $/kWh


#######Other inputs########
cycle_number = 1 #Equivalent cycle number for a year, only affects operation (the higher the number is the less effect there will be), set as now as I am not sure how the maximum sotrage capacity is determined and how the storage will be cycled
path_tankinator= r'enter tankinator path here'

#####Run calculation########
Compressed_gas_function.func(Wind_avai, H2_flow, cdratio, Energy_cost, cycle_number, path_tankinator)



########Outputs################

######Maximum equivalent storage capacity and duration
Compressed_gas_function.capacity_max   #This is the maximum equivalent H2 storage in kg
Compressed_gas_function.t_discharge_hr_max   #This is tha maximum storage duration in kg


###Parameters for capital cost fitting for optmizing capital cost
a_fit = Compressed_gas_function.a_fit
b_fit = Compressed_gas_function.b_fit



#Parameters for operational cost fitting for optmizing capital cost
a_op_fit = Compressed_gas_function.a_op_fit
b_op_fit = Compressed_gas_function.b_op_fit
c_op_fit = Compressed_gas_function.c_op_fit


