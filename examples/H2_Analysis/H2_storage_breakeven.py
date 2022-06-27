from pressure_vessel_model import Pressure_Vessel_Storage
from underground_pipe_storage import Underground_Pipe_Storage
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

"""pressure vessel storage"""
in_dict = dict()
out_dict = dict()
# static in analysis
in_dict['compressor_output_pressure'] = 250     #[bar]
# changes during analysis
H2_kg_storage_sweep = np.arange(1,500000, 10)                    #[kg]

pressure_vessel_npv = []

for i in range(len(H2_kg_storage_sweep)):
    in_dict['H2_storage_kg'] = H2_kg_storage_sweep[i]
    pressure_vessel = Pressure_Vessel_Storage(in_dict,out_dict)
    pressure_vessel.pressure_vessel_costs()
    pressure_vessel_npv_i = npf.npv(0.07,out_dict['pressure_vessel_annuals'])
    pressure_vessel_npv.append(pressure_vessel_npv_i)
    # print("Pressure Vessel Storage NPV: ", pressure_vessel_npv, "[USD]")
    
"""pipe storage"""
input_dict = dict()
output_dict = dict()
# static in analysis
input_dict['compressor_output_pressure'] = 100     #[bar]


pipe_storage_npv = []

for i in range(len(H2_kg_storage_sweep)):
    input_dict['H2_storage_kg'] = H2_kg_storage_sweep[i]
    pipe_storage = Underground_Pipe_Storage(input_dict,output_dict)
    pipe_storage.pipe_storage_costs()
    pipe_storage_npv_i = npf.npv(0.07,output_dict['pipe_storage_annuals'])
    pipe_storage_npv.append(pipe_storage_npv_i)
    # print("Pipe Storage NPV: ", pipe_storage_npv, "[USD]")

plt.plot(H2_kg_storage_sweep, pipe_storage_npv, label = 'Pipe storage npv')
plt.plot(H2_kg_storage_sweep, pressure_vessel_npv, label = 'Pressure vessel storage npv')
plt.xlabel('Hydrogen storage [kg]')
plt.ylabel('Storage npv [USD]')
plt.legend()
plt.show()

