import json
import os
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
from tools.powerflow import plant_grid, cable, power_flow_solvers
from tools.powerflow import visualization
from random import random

####Powerflow analysis in Colorado


### Make GIF from saved results
import imageio
images = []



# Load location information and generation profiles
with open('colorado_powerflow_profiles/power_plant_locations.json') as f:
  power_plant_details = json.load(f)

# Print plant names and locations
[print("Plant Name: {}, Lat: {}, Long: {}".format(plant['Location'], plant['Latitude'], plant['Longitude'])) for plant
 in power_plant_details]

# Add the generation profile to each plant and location
# 'Plant' in dict in list imported from json gives plant letter.
# Use this to determine which gen signal to load
plant_ids = [plant['Plant'] for plant in power_plant_details]
gen_file_names = ["results_{}.json".format(plant_id) for plant_id in plant_ids]
power_plant_details_new = []
for i, plant in enumerate(power_plant_details):
    with open("colorado_powerflow_profiles/{}".format(gen_file_names[i])) as f2:
        result = json.load(f2)
    plant['result'] = result
    plant['gen_wind'] = result['outputs']['Scenario']['Site']['Wind']['year_one_power_production_series_kw']
    plant['gen_solar'] = result['outputs']['Scenario']['Site']['PV']['year_one_power_production_series_kw']
    power_plant_details_new.append(plant)

power_plant_details = power_plant_details_new

# Transform lat/lon points to x,y in meters to provide cable run lengths
points = [(plant['Latitude'], plant['Longitude']) for plant in power_plant_details]
transformer = Transformer.from_crs(4326, 2100)
points_new = []
sum_x = 0
sum_y = 0
for pt in transformer.itransform(points):
    pt_new = pt
    points_new.append(pt_new)
    sum_x += pt_new[0]
    sum_y += pt_new[1]

mean_x, mean_y = (sum_x / len(points_new), sum_y / len(points_new))
points_demean = [tuple(np.subtract(point_new, (mean_x, mean_y))) for point_new in points_new]

# Connect node network
print(points_demean)
grid_connect_coords = np.array([[-90000, 0, 0]])
node_coordinates_list = [[int(x), int(y), 0] for x, y in points_demean]

node_coordinates = np.array(node_coordinates_list)
# node_labels = ['T{}'.format(node_num) for node_num in range(1, len(node_coordinates) + 1)] #Autogenerate node labels
node_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'HUB')
power_flow_grid_model.add_nodes(node_coordinates, node_labels)
cable1 = cable.CableByGeometry(1000 / 1000 ** 2)
node_connection_matrix = list()
# node_connection_matrix_hub = ['G', 'T1']
# # node_connection_matrix_turbines = [['T{}'.format(turb_node_label_num), 'T{}'.format(turb_node_label_num + 1)]
# #                                    for turb_node_label_num in range(1, len(node_coordinates))]
# node_connection_matrix_turbines = [['T1', 'T2'], ['T2', 'T3'], ['T3', 'T4'],
#                               ['T4', 'T5'], ['T5', 'T6'], ['T6', 'T7'], ['T7', 'T1']]
# node_connection_matrix.append(node_connection_matrix_hub)
# node_connection_matrix += node_connection_matrix_turbines
node_connection_matrix = [['HUB', 'A'], ['HUB', 'B'], ['HUB', 'C'], ['HUB', 'D'],
                              ['HUB', 'E'], ['HUB', 'F'], ['HUB', 'G']]
power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')

power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)

# Add summary details to dataframe
df_powerflow_summary = pd.DataFrame()
# hybrid_size_kw = 100
# df_powerflow_summary.hybrid_size_kw = hybrid_size_kw
df_powerflow_summary.node_labels = node_labels
df_powerflow_summary.cable = cable1
df_powerflow_summary.turbine_coords = node_coordinates

# Create lists to save details at each timestep
df_powerflow_details = pd.DataFrame()
gen_at_time_list = list()
P_turbs_list = list()
P_losses_list = list()
P_losses_pc_list = list()
s_NR_list = list()
i_NR_list = list()
v_NR_list = list()
comment_list = list()

# iterate through each gen timestep

for pstep in range(1, 100): # len(power_plant_details_new[0]['gen_wind'])):
    P_turbs = [[plant['gen_wind'][pstep]+ (random() * plant['gen_wind'][pstep] * 0.01e1j)] for plant in
               power_plant_details_new]
    if any(P_turbs) > 0:
        # power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'HUB')
        # power_flow_grid_model.add_nodes(node_coordinates, node_labels)
        # power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')
        power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)

        if pstep == 22:
            # Set some of the generation at different turbines to zero
            # P_turbs[3][0] = 0
            comment = '(Turbine 4, 5, 6, 7 set to zero)'
        elif pstep == 999:
            #  Break one of the connections:
            #  Node connection to node 11 gets removed here.
            new_node_connection_matrix = [['G', 'T1'], ['T1', 'T2'], ['T2', 'T3'], ['T3', 'T4'],
                                          ['T4', 'T5'], ['T5', 'T6'], ['T6', 'T7']]
            power_flow_grid_model.add_connections(new_node_connection_matrix, cable1, length_method='direct')
            comment = '(Removed link between nodes 10-11-12)'
        else:
            comment = ''

        P_turbs = np.array(P_turbs)

        # Solve power flow equations
        s_NR, v_NR, i_NR = power_flow_solvers.Netwon_Raphson_method(
            power_flow_grid_model.admittance_matrix_pu,
            P_turbs / power_flow_grid_model.nominal_properties['power'],
            max_iterations=20,
            quiet=False)
        print('Voltages:', v_NR * power_flow_grid_model.nominal_properties['voltage'])
        s_found = s_NR * power_flow_grid_model.nominal_properties['power']
        P_losses = np.sum(s_found)
        P_losses_pc = 100 * np.real(P_losses) / np.sum(np.real(P_turbs)) + \
                      100j * np.imag(P_losses) / np.sum(np.imag(P_turbs))
        print('Losses (real and reactive):', P_losses_pc, '%.')
        print("Timestep".format(pstep))
        P_turbs_list.append(P_turbs)
        P_losses_list.append(P_losses)
        P_losses_pc_list.append(P_losses_pc)
        s_NR_list.append(s_NR)
        i_NR_list.append(i_NR)
        v_NR_list.append(v_NR)
        comment_list.append(comment)

        # Visualize
        real_power = np.real(P_turbs)
        real_power = [x[0] for x in real_power]
        real_power.insert(0,0)
        real_power = np.array(real_power)
        ax, plt = visualization.grid_layout(power_flow_grid_model)
        visualization.overlay_quantity(power_flow_grid_model, real_power, ax, 'Real Power (kW)',
                                       'Real Power at Timestep {} {}'.format(pstep, comment))
        plotname = '{}_Real Power{}'.format(pstep, '.jpg')
        plt.savefig(os.path.join('results', plotname))
        plt.close()

        real_voltage = np.real(v_NR)
        real_voltage = [abs(x[0]) for x in real_voltage]
        real_voltage = np.array(real_voltage)
        ax, plt = visualization.grid_layout(power_flow_grid_model)
        visualization.overlay_quantity(power_flow_grid_model, real_voltage, ax, 'Real Voltage (V)',
                                       'Real Voltage at Timestep {} {}'.format(pstep, comment))
        plotname = '{}_Real Voltage{}'.format(pstep, '.jpg')
        plt.savefig(os.path.join('results', plotname))
        plt.close()

        # reactive_power = np.imag(P_turbs)
        # reactive_power = [x[0] for x in reactive_power]
        # reactive_power = np.array(reactive_power)
        # ax, plt = visualization.grid_layout(power_flow_grid_model)
        # visualization.overlay_quantity(power_flow_grid_model, reactive_power, ax, 'Reactive Power (kW)',
        #                                'Reactive Power at Timestep {} {}'.format(pstep, comment))
        # plotname = '{}_Reactive Power{}'.format(pstep, '.jpg')
        # plt.savefig(os.path.join('results', plotname))
        # plt.close()



        # reactive_voltage = np.imag(v_NR)
        # reactive_voltage = [abs(x[0]) for x in reactive_voltage]
        # reactive_voltage = np.array(reactive_voltage)
        # ax, plt = visualization.grid_layout(power_flow_grid_model)
        # visualization.overlay_quantity(power_flow_grid_model, reactive_voltage, ax, 'Reactive Voltage (V)',
        #                                'Reactive Voltage at Timestep {} {}'.format(pstep, comment))
        # plotname = '{}_Reactive Voltage{}'.format(pstep, '.jpg')
        # plt.savefig(os.path.join('results', plotname))
        # plt.close()

    else:
        P_turbs = 0
        P_losses = 0
        P_losses_pc = 0
        s_NR = 0
        i_NR = 0
        v_NR = 0
        comment = 'No generation from turbines'
        P_turbs_list.append(P_turbs)
        P_losses_list.append(P_losses)
        P_losses_pc_list.append(P_losses_pc)
        s_NR_list.append(s_NR)
        i_NR_list.append(i_NR)
        v_NR_list.append(v_NR)
        comment_list.append(comment)

powerflow_dict = {'Comment': comment_list, 'P_turbs': P_turbs_list,
                  'P_losses': P_losses_list, 'P_losses_percent': P_losses_pc_list,
                  's_NR': s_NR_list, 'i_NR': i_NR_list, 'v_NR': v_NR_list}

df_powerflow_details = pd.DataFrame(powerflow_dict)
df_powerflow_details.to_csv('powerflow_details.csv')


relevant_path = 'results'
included_extensions = ['Power.jpg']
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

import re
file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
for filename in file_names:
    images.append(imageio.imread(os.path.join('results', filename)))
imageio.mimsave('RealPower.gif', images)