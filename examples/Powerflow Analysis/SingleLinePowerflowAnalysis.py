import json
import os
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
from tools.powerflow import plant_grid, cable, power_flow_solvers
from tools.powerflow import visualization
from random import random
from operator import add

####Powerflow analysis in Colorado


### Make GIF from saved results
import imageio
images = []
all_points = [(3700,0)]
grid_connect_coords = np.array([[0, 0, 0]])
node_coordinates_list = [[int(x), int(y), 0] for x, y in all_points]

node_coordinates = np.array(node_coordinates_list)
# node_labels = ['T{}'.format(node_num) for node_num in range(1, len(node_coordinates) + 1)] #Autogenerate node labels
node_labels = ['Only Node']
power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'HUB')
power_flow_grid_model.add_nodes(node_coordinates, node_labels)
cable1 = cable.CableByGeometry(cross_sectional_area=.0005,
                       line_type='underground',
                       phase_separation_distance=0.1, # defaulting to 100mm
                       resistivity=0.0808/1000, # Ohms/m -> Ohms/km #1.724e-8, # Copper at 20C
                       skin_correction_factor=1.02,
                       relative_permittivity=2.3,
                       relative_permeability=1)
node_connection_matrix = list()
node_connection_matrix = [['HUB', 'Only Node']]
power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')
power_flow_grid_model.assign_nominal_quantities(400*33000, 33e3)

# Add summary details to dataframe
df_powerflow_summary = pd.DataFrame()
df_powerflow_summary.node_labels = node_labels
df_powerflow_summary.cable = cable1
df_powerflow_summary.turbine_coords = node_coordinates

# Create lists to save details at each timestep
df_powerflow_details = pd.DataFrame()
gen_at_time_list = list()
P_nodes_list = list()
P_losses_list = list()
P_losses_pc_list = list()
s_NR_list = list()
i_NR_list = list()
v_NR_list = list()
comment_list = list()

P_nodes = [[(413*33) + (1000 * 0.01e1j)]]
P_nodes = np.array(P_nodes)

# Solve power flow equations
s_NR, v_NR, i_NR = power_flow_solvers.Netwon_Raphson_method(
    power_flow_grid_model.admittance_matrix_pu,
    P_nodes / power_flow_grid_model.nominal_properties['power'],
    max_iterations=20,
    quiet=False)
print('Voltages:', v_NR * power_flow_grid_model.nominal_properties['voltage'])
s_found = s_NR * power_flow_grid_model.nominal_properties['power']
P_losses = np.sum(s_found)
P_losses_pc = 100 * np.real(P_losses) / np.sum(np.real(P_nodes)) + \
              100j * np.imag(P_losses) / np.sum(np.imag(P_nodes))
print(np.real(P_losses_pc))
loss_text = ['Losses (real and reactive):{:.2f} + {:.2f}'.format(np.real(P_losses_pc), np.imag(P_losses_pc))]
# loss_text_voltage = ['Losses (real and reactive):{:.2f} + {:.2f}'.format(np.real(P_losses_pc), np.imag(P_losses_pc))]
print(loss_text)

P_nodes_list.append(P_nodes)
P_losses_list.append(P_losses)
P_losses_pc_list.append(P_losses_pc)
s_NR_list.append(s_NR)
i_NR_list.append(i_NR)
v_NR_list.append(v_NR)

# Visualize
real_power = np.real(P_nodes)
real_power = [x[0] for x in real_power]
real_power.insert(0, 0)
real_power = np.array(real_power)
loads = np.array((random() * (real_power/2)))

ax, plt = visualization.grid_layout(power_flow_grid_model)
visualization.overlay_quantity(power_flow_grid_model, loads, ax, 'Real Power (kW)',
                               'Real Power at Timestep {} {} \n{}'.format('', '', loss_text), color=[1, 0, 0], offset=0)

plotname = '{}_Real Power{}'.format('', '.jpg')
plt.savefig(os.path.join('../H2 Analysis/results', plotname))
plt.show()
# plt.close()

real_voltage = np.real(v_NR)
real_voltage = [abs(x[0]) for x in real_voltage]
real_voltage = np.array(real_voltage)
ax, plt = visualization.grid_layout(power_flow_grid_model)
visualization.overlay_quantity(power_flow_grid_model, real_voltage, ax, 'Real Voltage (V)',
                               'Real Voltage at Timestep {} {} \n{}'.format('','','loss_text_voltage'), color=[1, 0, 0], offset=0)
plotname = '{}_Real Voltage{}'.format('', '.jpg')
plt.savefig(os.path.join('../H2 Analysis/results', plotname))
plt.show()
# plt.close()


# powerflow_dict = {'Comment': comment_list, 'P_nodes': P_nodes_list,
#                   'P_losses': P_losses_list, 'P_losses_percent': P_losses_pc_list,
#                   's_NR': s_NR_list, 'i_NR': i_NR_list, 'v_NR': v_NR_list}
#
# df_powerflow_details = pd.DataFrame(powerflow_dict)
# df_powerflow_details.to_csv('powerflow_details.csv')
#
#
# relevant_path = '../H2 Analysis/results'
# included_extensions = ['Power.jpg']
# file_names = [fn for fn in os.listdir(relevant_path)
#               if any(fn.endswith(ext) for ext in included_extensions)]
#
# import re
# file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
# for filename in file_names:
#     images.append(imageio.imread(os.path.join('../H2 Analysis/results', filename)))
# imageio.mimsave('RealPower.gif', images)