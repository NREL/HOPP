import json
import os
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
# from tools.powerflow import plant_grid, cable, power_flow_solvers
from tools.powerflow import visualization
from random import random
from operator import add
import pandapower as pp
import pandapower.converter as pc
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly

#### Powerflow analysis in Colorado using Pandapower ####

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
point_load = (points[3][0], points[3][1]+.18)
points.append(point_load)
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
# Create an empty Pandapower network
net = pp.create_empty_network()
pp_2k_net = pc.from_mpc('tamu2k_grid/tamu2k.mat', f_hz=50)
# Create the busses
#Todo: Automate the bus creation - easy, just requires keeping track of the bus number for assigning things later
b1 = pp.create_bus(net, vn_kv=20., name=power_plant_details_new[0]['Name'], geodata=points_new[0])
b2 = pp.create_bus(net, vn_kv=20., name=power_plant_details_new[1]['Name'], geodata=points_new[1])
b3 = pp.create_bus(net, vn_kv=20., name=power_plant_details_new[2]['Name'], geodata=points_new[2])
b4 = pp.create_bus(net, vn_kv=20., name=power_plant_details_new[3]['Name'], geodata=points_new[3])
b5 = pp.create_bus(net, vn_kv=20., name=power_plant_details_new[4]['Name'], geodata=points_new[4])
b6 = pp.create_bus(net, vn_kv=20., name=power_plant_details_new[5]['Name'], geodata=points_new[5])
b7 = pp.create_bus(net, vn_kv=20., name=power_plant_details_new[6]['Name'], geodata=points_new[6])
b_load = pp.create_bus(net, vn_kv=20, name="Load Centre", geodata=points_new[7])

# Create the external grid bus element
pp.create_ext_grid(net, bus=b_load, vm_pu=1.02, name="Grid Connection")

# Create the loads
pp.create_load(net, bus=b_load, p_mw=3, q_mvar=.50, name="Load")
pp.create_load(net, bus=b5, p_mw=3, q_mvar=.50, name="Load")

# Create a transformer
# tid = pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV")
# tid = pp.create_transformer_from_parameters(net, sn_mva=0.4,
#                                             hv_bus=b1, lv_bus=b2,
#                                             vn_hv_kv=20., vn_lv_kv=0.4,
#                                             vk_percent=6., vkr_percent=1.425,
#                                             i0_percent=0.3375, pfe_kw=1.35,
#                                             name="Trafo")

# Create a line
lid = pp.create_line(net, from_bus=b1, to_bus=b3, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")

lid = pp.create_line(net, from_bus=b3, to_bus=b5, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")
lid = pp.create_line(net, from_bus=b5, to_bus=b6, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")
lid = pp.create_line(net, from_bus=b6, to_bus=b7, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")
lid = pp.create_line(net, from_bus=b7, to_bus=b1, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")

lid = pp.create_line(net, from_bus=b2, to_bus=b4, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")
lid = pp.create_line(net, from_bus=b4, to_bus=b_load, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")

lid = pp.create_line(net, from_bus=b2, to_bus=b1, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")

# Run a Pandapower analysis
#TODO Remove after running t2k
# pp.runpp(pp_2k_net)
latlondata_tu2k = pd.read_csv('tamu2k_grid/tamu2k_latlon.csv')
list_of_lat_lon_info_tuples = list(latlondata_tu2k.itertuples(index=False))
lat_lon_tuples = [(lis[2],lis[3]) for lis in list_of_lat_lon_info_tuples]
pp.create_bus(pp_2k_net, geodata=lat_lon_tuples, vn_kv=20)
# pp.plotting.create_generic_coordinates(pp_2k_net, mg=None, library='igraph', respect_switches=False, geodata_table='bus_geodata', buses=None, overwrite=False) #generates random coords to test plotting
simple_plotly(pp_2k_net, on_map=False, projection='epsg:2100')
# pf_res_plotly(pp_2k_net, on_map=True, projection='epsg:2100')


# pp.runpp(net)
from pandapower.networks import mv_oberrhein # Sample scenario to test plotting
# pp.plotting.create_generic_coordinates(net, mg=None, library='igraph', respect_switches=False, geodata_table='bus_geodata', buses=None, overwrite=False) #generates random coords to test plotting
# pp.plotting.plotly.mapbox_plot.set_mapbox_token('pk.eyJ1IjoiYmlnY2hyb21lIiwiYSI6ImNrbmtuanM0cDBhdHMyb2x0eG4wYTRyMWsifQ.0-FshnrGMfmd-TwqEeclAA')

# Simple Network Plot
# simple_plotly(net, on_map=True, projection='epsg:2100')
# Losses and Voltages Plot
# pf_res_plotly(net, on_map=True, projection='epsg:2100')


#print element tables
print("-------------------")
print("  ELEMENT TABLES   ")
print("-------------------")

print("net.bus")
print(net.bus)

print("\n net.trafo")
print(net.trafo)

print("\n net.line")
print(net.line)

print("\n net.load")
print(net.load)

print("\n net.ext_grid")
print(net.ext_grid)

#print result tables
print("\n-------------------")
print("   RESULT TABLES   ")
print("-------------------")

print("net.res_bus")
print(net.res_bus)

print("\n net.res_trafo")
print(net.res_trafo)

print("\n net.res_line")
print(net.res_line)

print("\n net.res_load")
print(net.res_load)

print("\n net.res_ext_grid")
print(net.res_ext_grid)

print('And some more time here')

# node_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'HUB')
# power_flow_grid_model.add_nodes(node_coordinates, node_labels)
# cable1 = cable.CableByGeometry(100 / 1000 ** 2)
# node_connection_matrix = list()
# # node_connection_matrix_hub = ['G', 'T1']
# # # node_connection_matrix_turbines = [['T{}'.format(turb_node_label_num), 'T{}'.format(turb_node_label_num + 1)]
# # #                                    for turb_node_label_num in range(1, len(node_coordinates))]
# # node_connection_matrix_turbines = [['T1', 'T2'], ['T2', 'T3'], ['T3', 'T4'],
# #                               ['T4', 'T5'], ['T5', 'T6'], ['T6', 'T7'], ['T7', 'T1']]
# # node_connection_matrix.append(node_connection_matrix_hub)
# # node_connection_matrix += node_connection_matrix_turbines
# node_connection_matrix = [['HUB', 'A'], ['HUB', 'B'], ['HUB', 'C'], ['HUB', 'D'],
#                               ['HUB', 'E'], ['HUB', 'F'], ['HUB', 'G']]
# power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')
#
# power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)
#
# # Add summary details to dataframe
# df_powerflow_summary = pd.DataFrame()
# df_powerflow_summary.node_labels = node_labels
# df_powerflow_summary.cable = cable1
# df_powerflow_summary.turbine_coords = node_coordinates
#
# # Create lists to save details at each timestep
# df_powerflow_details = pd.DataFrame()
# gen_at_time_list = list()
# P_nodes_list = list()
# P_losses_list = list()
# P_losses_pc_list = list()
# s_NR_list = list()
# i_NR_list = list()
# v_NR_list = list()
# comment_list = list()
#
# # iterate through each gen timestep
#
# for pstep in range(1, 100): # len(power_plant_details_new[0]['gen_wind'])):
#     P_nodes_wind = [[plant['gen_wind'][pstep] + (random() * plant['gen_wind'][pstep] * 0.01e1j)] for plant in
#                power_plant_details_new]
#     P_nodes_solar = [[plant['gen_solar'][pstep] + (random() * plant['gen_solar'][pstep] * 0.01e1j)] for plant in
#                power_plant_details_new]
#     P_nodes = [[plant['gen_wind'][pstep] + plant['gen_solar'][pstep] +
#                 (random() * (plant['gen_wind'][pstep]+plant['gen_solar'][pstep]) * 0.01e1j)] for plant in
#                power_plant_details_new]
#
#     if any(P_nodes) > 0:
#         # power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'HUB')
#         # power_flow_grid_model.add_nodes(node_coordinates, node_labels)
#         # power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')
#         power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)
#
#         if pstep == 22:
#             # Set some of the generation at different turbines to zero
#             # P_turbs[3][0] = 0
#             new_node_connection_matrix = [['HUB', 'A'], ['HUB', 'C'], ['HUB', 'D'],
#                                       ['HUB', 'E'], ['HUB', 'F'], ['HUB', 'G']]
#             # power_flow_grid_model.add_connections(new_node_connection_matrix, cable1, length_method='direct')
#             # comment = '(Broken link for Node B)'
#         elif pstep == 999:
#             #  Break one of the connections:
#             #  Node connection to node 11 gets removed here.
#             comment = 'Do Something Else'
#         else:
#             comment = ''
#
#         P_nodes = np.array(P_nodes)
#
#         # Solve power flow equations
#         s_NR, v_NR, i_NR = power_flow_solvers.Netwon_Raphson_method(
#             power_flow_grid_model.admittance_matrix_pu,
#             P_nodes / power_flow_grid_model.nominal_properties['power'],
#             max_iterations=20,
#             quiet=False)
#         print('Voltages:', v_NR * power_flow_grid_model.nominal_properties['voltage'])
#         s_found = s_NR * power_flow_grid_model.nominal_properties['power']
#         P_losses = np.sum(s_found)
#         P_losses_pc = 100 * np.real(P_losses) / np.sum(np.real(P_nodes)) + \
#                       100j * np.imag(P_losses) / np.sum(np.imag(P_nodes))
#         print(np.real(P_losses_pc))
#         loss_text = ['Losses (real and reactive):{:.2f} + {:.2f}'.format(np.real(P_losses_pc), np.imag(P_losses_pc))]
#         print(loss_text)
#         print("Timestep".format(pstep))
#         P_nodes_list.append(P_nodes)
#         P_losses_list.append(P_losses)
#         P_losses_pc_list.append(P_losses_pc)
#         s_NR_list.append(s_NR)
#         i_NR_list.append(i_NR)
#         v_NR_list.append(v_NR)
#         comment_list.append(comment)
#
#         # Visualize
#         real_power = np.real(P_nodes)
#         real_power = [x[0] for x in real_power]
#         real_power.insert(0, 0)
#         real_power = np.array(real_power)
#         loads = np.array((random() * (real_power/2)))
#         wind_power = np.real(P_nodes_wind)
#
#         wind_power = np.array(np.real(P_nodes_wind))
#         wind_power = [x[0] for x in wind_power]
#         wind_power = np.array(wind_power)
#         solar_power = np.array(np.real(P_nodes_solar))
#         solar_power = [x[0] for x in solar_power]
#         solar_power = np.array(solar_power)
#
#         ax, plt = visualization.grid_layout(power_flow_grid_model)
#         visualization.overlay_quantity(power_flow_grid_model, loads, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[1, 0, 0], offset=2400)
#         visualization.overlay_quantity(power_flow_grid_model, real_power, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[0, 1, 0], offset=0)
#         visualization.overlay_quantity(power_flow_grid_model, wind_power, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[0, 0, 1], offset=4800)
#         visualization.overlay_quantity(power_flow_grid_model, solar_power, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[1, 0.75, 0], offset=6200)
#
#         plotname = '{}_Real Power{}'.format(pstep, '.jpg')
#         plt.savefig(os.path.join('../H2 Analysis/results', plotname))
#         plt.close()
#
#         real_voltage = np.real(v_NR)
#         real_voltage = [abs(x[0]) for x in real_voltage]
#         real_voltage = np.array(real_voltage)
#         ax, plt = visualization.grid_layout(power_flow_grid_model)
#         visualization.overlay_quantity(power_flow_grid_model, real_voltage, ax, 'Real Voltage (V)',
#                                        'Real Voltage at Timestep {} {}'.format(pstep, comment), color=[1, 0, 0], offset=2400)
#         plotname = '{}_Real Voltage{}'.format(pstep, '.jpg')
#         plt.savefig(os.path.join('../H2 Analysis/results', plotname))
#         plt.close()
#
#     else:
#         P_nodes = 0
#         P_losses = 0
#         P_losses_pc = 0
#         s_NR = 0
#         i_NR = 0
#         v_NR = 0
#         comment = 'No generation from turbines'
#         P_nodes_list.append(P_nodes)
#         P_losses_list.append(P_losses)
#         P_losses_pc_list.append(P_losses_pc)
#         s_NR_list.append(s_NR)
#         i_NR_list.append(i_NR)
#         v_NR_list.append(v_NR)
#         comment_list.append(comment)
#
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
# imageio.mimsave('RealPower.gif', images)node_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'HUB')
# power_flow_grid_model.add_nodes(node_coordinates, node_labels)
# cable1 = cable.CableByGeometry(100 / 1000 ** 2)
# node_connection_matrix = list()
# # node_connection_matrix_hub = ['G', 'T1']
# # # node_connection_matrix_turbines = [['T{}'.format(turb_node_label_num), 'T{}'.format(turb_node_label_num + 1)]
# # #                                    for turb_node_label_num in range(1, len(node_coordinates))]
# # node_connection_matrix_turbines = [['T1', 'T2'], ['T2', 'T3'], ['T3', 'T4'],
# #                               ['T4', 'T5'], ['T5', 'T6'], ['T6', 'T7'], ['T7', 'T1']]
# # node_connection_matrix.append(node_connection_matrix_hub)
# # node_connection_matrix += node_connection_matrix_turbines
# node_connection_matrix = [['HUB', 'A'], ['HUB', 'B'], ['HUB', 'C'], ['HUB', 'D'],
#                               ['HUB', 'E'], ['HUB', 'F'], ['HUB', 'G']]
# power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')
#
# power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)
#
# # Add summary details to dataframe
# df_powerflow_summary = pd.DataFrame()
# df_powerflow_summary.node_labels = node_labels
# df_powerflow_summary.cable = cable1
# df_powerflow_summary.turbine_coords = node_coordinates
#
# # Create lists to save details at each timestep
# df_powerflow_details = pd.DataFrame()
# gen_at_time_list = list()
# P_nodes_list = list()
# P_losses_list = list()
# P_losses_pc_list = list()
# s_NR_list = list()
# i_NR_list = list()
# v_NR_list = list()
# comment_list = list()
#
# # iterate through each gen timestep
#
# for pstep in range(1, 100): # len(power_plant_details_new[0]['gen_wind'])):
#     P_nodes_wind = [[plant['gen_wind'][pstep] + (random() * plant['gen_wind'][pstep] * 0.01e1j)] for plant in
#                power_plant_details_new]
#     P_nodes_solar = [[plant['gen_solar'][pstep] + (random() * plant['gen_solar'][pstep] * 0.01e1j)] for plant in
#                power_plant_details_new]
#     P_nodes = [[plant['gen_wind'][pstep] + plant['gen_solar'][pstep] +
#                 (random() * (plant['gen_wind'][pstep]+plant['gen_solar'][pstep]) * 0.01e1j)] for plant in
#                power_plant_details_new]
#
#     if any(P_nodes) > 0:
#         # power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'HUB')
#         # power_flow_grid_model.add_nodes(node_coordinates, node_labels)
#         # power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')
#         power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)
#
#         if pstep == 22:
#             # Set some of the generation at different turbines to zero
#             # P_turbs[3][0] = 0
#             new_node_connection_matrix = [['HUB', 'A'], ['HUB', 'C'], ['HUB', 'D'],
#                                       ['HUB', 'E'], ['HUB', 'F'], ['HUB', 'G']]
#             # power_flow_grid_model.add_connections(new_node_connection_matrix, cable1, length_method='direct')
#             # comment = '(Broken link for Node B)'
#         elif pstep == 999:
#             #  Break one of the connections:
#             #  Node connection to node 11 gets removed here.
#             comment = 'Do Something Else'
#         else:
#             comment = ''
#
#         P_nodes = np.array(P_nodes)
#
#         # Solve power flow equations
#         s_NR, v_NR, i_NR = power_flow_solvers.Netwon_Raphson_method(
#             power_flow_grid_model.admittance_matrix_pu,
#             P_nodes / power_flow_grid_model.nominal_properties['power'],
#             max_iterations=20,
#             quiet=False)
#         print('Voltages:', v_NR * power_flow_grid_model.nominal_properties['voltage'])
#         s_found = s_NR * power_flow_grid_model.nominal_properties['power']
#         P_losses = np.sum(s_found)
#         P_losses_pc = 100 * np.real(P_losses) / np.sum(np.real(P_nodes)) + \
#                       100j * np.imag(P_losses) / np.sum(np.imag(P_nodes))
#         print(np.real(P_losses_pc))
#         loss_text = ['Losses (real and reactive):{:.2f} + {:.2f}'.format(np.real(P_losses_pc), np.imag(P_losses_pc))]
#         print(loss_text)
#         print("Timestep".format(pstep))
#         P_nodes_list.append(P_nodes)
#         P_losses_list.append(P_losses)
#         P_losses_pc_list.append(P_losses_pc)
#         s_NR_list.append(s_NR)
#         i_NR_list.append(i_NR)
#         v_NR_list.append(v_NR)
#         comment_list.append(comment)
#
#         # Visualize
#         real_power = np.real(P_nodes)
#         real_power = [x[0] for x in real_power]
#         real_power.insert(0, 0)
#         real_power = np.array(real_power)
#         loads = np.array((random() * (real_power/2)))
#         wind_power = np.real(P_nodes_wind)
#
#         wind_power = np.array(np.real(P_nodes_wind))
#         wind_power = [x[0] for x in wind_power]
#         wind_power = np.array(wind_power)
#         solar_power = np.array(np.real(P_nodes_solar))
#         solar_power = [x[0] for x in solar_power]
#         solar_power = np.array(solar_power)
#
#         ax, plt = visualization.grid_layout(power_flow_grid_model)
#         visualization.overlay_quantity(power_flow_grid_model, loads, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[1, 0, 0], offset=2400)
#         visualization.overlay_quantity(power_flow_grid_model, real_power, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[0, 1, 0], offset=0)
#         visualization.overlay_quantity(power_flow_grid_model, wind_power, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[0, 0, 1], offset=4800)
#         visualization.overlay_quantity(power_flow_grid_model, solar_power, ax, 'Real Power (kW)',
#                                        'Real Power at Timestep {} {} \n{}'.format(pstep, comment, loss_text), color=[1, 0.75, 0], offset=6200)
#
#         plotname = '{}_Real Power{}'.format(pstep, '.jpg')
#         plt.savefig(os.path.join('../H2 Analysis/results', plotname))
#         plt.close()
#
#         real_voltage = np.real(v_NR)
#         real_voltage = [abs(x[0]) for x in real_voltage]
#         real_voltage = np.array(real_voltage)
#         ax, plt = visualization.grid_layout(power_flow_grid_model)
#         visualization.overlay_quantity(power_flow_grid_model, real_voltage, ax, 'Real Voltage (V)',
#                                        'Real Voltage at Timestep {} {}'.format(pstep, comment), color=[1, 0, 0], offset=2400)
#         plotname = '{}_Real Voltage{}'.format(pstep, '.jpg')
#         plt.savefig(os.path.join('../H2 Analysis/results', plotname))
#         plt.close()
#
#     else:
#         P_nodes = 0
#         P_losses = 0
#         P_losses_pc = 0
#         s_NR = 0
#         i_NR = 0
#         v_NR = 0
#         comment = 'No generation from turbines'
#         P_nodes_list.append(P_nodes)
#         P_losses_list.append(P_losses)
#         P_losses_pc_list.append(P_losses_pc)
#         s_NR_list.append(s_NR)
#         i_NR_list.append(i_NR)
#         v_NR_list.append(v_NR)
#         comment_list.append(comment)
#
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