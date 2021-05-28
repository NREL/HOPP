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
import imageio

#### Powerflow analysis using Pandapower ####

### TODO: Make GIF from saved results
images = []

# Create Pandapower network from .mat file
pp_2k_net = pc.from_mpc('tamu2k_grid/TAMU2K_new_ver1.mat', f_hz=50, validate_conversion=False)
# pp_2k_net = pp.converter.pypower.from_ppc('tamu2k_grid/TAMU2K_new_ver2.mat', f_hz=50)

# Run a Pandapower analysis
result = pp.runopp(pp_2k_net, max_iteration=100, algorithm='nr')
latlondata_tu2k = pd.read_csv('tamu2k_grid/tamu2k_latlon.csv')
list_of_lat_lon_info_tuples = list(latlondata_tu2k.itertuples(index=False))
lat_lon_tuples = [(lis[2], lis[3]) for lis in list_of_lat_lon_info_tuples]
# Transform Texas lat/lon points to x,y in meters
transformer = Transformer.from_crs(4326, 2100)
points_new_texas = []
for pt in transformer.itransform(lat_lon_tuples):
    pt_new = pt
    points_new_texas.append(pt_new)

bus_geodata_texas = pd.DataFrame(points_new_texas, columns=['x', 'y'])
coords = np.zeros(len(points_new_texas))  # TODO: Should be NaN, change if issues
bus_geodata_texas['coords'] = coords

# Add Geodata to network
pp_2k_net.bus_geodata = bus_geodata_texas
# pp.plotting.create_generic_coordinates(pp_2k_net, mg=None, library='igraph', respect_switches=False, geodata_table='bus_geodata', buses=None, overwrite=False) #generates random coords to test plotting
# # simple_plotly(pp_2k_net, on_map=False, projection='epsg:2100')
# # pf_res_plotly(pp_2k_net, on_map=True, projection='epsg:2100')
#
#
# # Run Through timeseries
# pp_2k_net.gen['p_mw'] = np.zeros(2000)
# pp_2k_net.load['p_mw'] = np.zeros(2000)
# pp_2k_net.load['q_mvar'] = np.zeros(2000)

print('Gen and load values set')
# Simple Network Plot
# simple_plotly(net, on_map=True, projection='epsg:2100')
# Losses and Voltages Plot
# pf_res_plotly(net, on_map=True, projection='epsg:2100')
# pp.plotting.create_generic_coordinates(net, mg=None, library='igraph', respect_switches=False, geodata_table='bus_geodata', buses=None, overwrite=False) #generates random coords to test plotting
# pp_2k_net.plotting.plotly.mapbox_plot.set_mapbox_token('pk.eyJ1IjoiYmlnY2hyb21lIiwiYSI6ImNrbmtuanM0cDBhdHMyb2x0eG4wYTRyMWsifQ.0-FshnrGMfmd-TwqEeclAA')

#print element tables
print("-------------------")
print("  ELEMENT TABLES   ")
print("-------------------")

print("net.bus")
print(pp_2k_net.bus)

print("\n net.trafo")
print(pp_2k_net.trafo)

print("\n net.line")
print(pp_2k_net.line)

print("\n net.load")
print(pp_2k_net.load)

print("\n net.ext_grid")
print(pp_2k_net.ext_grid)

#print result tables
print("\n-------------------")
print("   RESULT TABLES   ")
print("-------------------")

print("net.res_bus")
print(pp_2k_net.res_bus)

print("\n net.res_trafo")
print(pp_2k_net.res_trafo)

print("\n net.res_line")
print(pp_2k_net.res_line)

print("\n net.res_load")
print(pp_2k_net.res_load)

print("\n net.res_ext_grid")
print(pp_2k_net.res_ext_grid)


